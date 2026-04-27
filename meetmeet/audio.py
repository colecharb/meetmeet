from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

WHISPER_RATE = 16000


@dataclass
class DeviceSelection:
    mic_index: int | None
    mic_name: str | None
    blackhole_index: int | None
    blackhole_name: str | None


def find_devices() -> DeviceSelection:
    """Return the default mic and any BlackHole input device."""
    devices = sd.query_devices()
    default_in = sd.default.device[0] if sd.default.device else None
    mic_index = None
    mic_name = None
    if default_in is not None and 0 <= default_in < len(devices):
        d = devices[default_in]
        if d.get("max_input_channels", 0) > 0:
            mic_index = default_in
            mic_name = d.get("name")

    bh_index = None
    bh_name = None
    for i, d in enumerate(devices):
        name = d.get("name", "")
        if "blackhole" in name.lower() and d.get("max_input_channels", 0) >= 2:
            bh_index = i
            bh_name = name
            break

    return DeviceSelection(mic_index, mic_name, bh_index, bh_name)


def probe_mic_audio(device: int, sample_rate: int, seconds: float = 0.5) -> bool:
    """Open a brief input stream and check that it produces non-silent samples.
    Used to detect the macOS mic-permission-denied case where streams open but yield zeros."""
    try:
        data = sd.rec(
            int(seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            device=device,
            dtype="float32",
            blocking=True,
        )
    except Exception:
        return False
    return bool(np.max(np.abs(data)) > 1e-5)


class _StreamSource:
    """Wraps a single sd.InputStream into a thread-safe mono float32 deque."""

    def __init__(self, device_index: int, sample_rate: int, blocksize: int = 2048):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self._lock = threading.Lock()
        self._buffer: deque[np.ndarray] = deque()
        info = sd.query_devices(device_index)
        self._channels = max(1, min(int(info.get("max_input_channels", 1)), 2))
        self._stream: sd.InputStream | None = None

    def _callback(self, indata: np.ndarray, frames, time_info, status) -> None:
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata.mean(axis=1)
        else:
            mono = indata.reshape(-1)
        with self._lock:
            self._buffer.append(mono.astype(np.float32, copy=True))

    def start(self) -> None:
        self._stream = sd.InputStream(
            device=self.device_index,
            samplerate=self.sample_rate,
            channels=self._channels,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def drain(self, max_samples: int) -> np.ndarray:
        """Pop up to max_samples worth of audio from the buffer."""
        out: list[np.ndarray] = []
        total = 0
        with self._lock:
            while self._buffer and total < max_samples:
                block = self._buffer.popleft()
                if total + len(block) > max_samples:
                    take = max_samples - total
                    out.append(block[:take])
                    leftover = block[take:]
                    if len(leftover) > 0:
                        self._buffer.appendleft(leftover)
                    total += take
                else:
                    out.append(block)
                    total += len(block)
        if not out:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out)

    def buffered_samples(self) -> int:
        with self._lock:
            return sum(len(b) for b in self._buffer)


@dataclass
class TranscribeChunk:
    chunk_id: int
    start_wall: datetime
    audio: np.ndarray  # float32 mono at WHISPER_RATE


class AudioPipeline:
    """Mic + (optional) BlackHole -> mixer -> chunked, resampled queue ready for STT."""

    def __init__(
        self,
        mic_index: int | None,
        blackhole_index: int | None,
        capture_rate: int,
        chunk_seconds: float,
        chunk_overlap_seconds: float,
        out_queue: "queue.Queue[TranscribeChunk]",
        on_overflow: Callable[[], None] | None = None,
    ):
        if mic_index is None and blackhole_index is None:
            raise RuntimeError("no audio input devices available")
        self.capture_rate = capture_rate
        self.chunk_seconds = chunk_seconds
        self.chunk_overlap_seconds = chunk_overlap_seconds
        self.out_queue = out_queue
        self.on_overflow = on_overflow
        self._mic = _StreamSource(mic_index, capture_rate) if mic_index is not None else None
        self._bh = _StreamSource(blackhole_index, capture_rate) if blackhole_index is not None else None
        self._stop = threading.Event()
        self._mixer_thread: threading.Thread | None = None
        self._chunk_id = 0

    @property
    def sources(self) -> list[str]:
        s = []
        if self._mic is not None:
            s.append("mic")
        if self._bh is not None:
            s.append("blackhole")
        return s

    def start(self) -> None:
        if self._mic is not None:
            self._mic.start()
        if self._bh is not None:
            self._bh.start()
        self._mixer_thread = threading.Thread(target=self._run_mixer, daemon=True)
        self._mixer_thread.start()

    def stop(self, flush: bool = True) -> None:
        self._stop.set()
        if self._mixer_thread is not None:
            self._mixer_thread.join(timeout=self.chunk_seconds + 2)
        if self._mic is not None:
            self._mic.stop()
        if self._bh is not None:
            self._bh.stop()
        if flush:
            self._flush_remaining()

    def _run_mixer(self) -> None:
        target_per_chunk = int((self.chunk_seconds + self.chunk_overlap_seconds) * self.capture_rate)
        poll_interval = 0.25
        while not self._stop.is_set():
            time.sleep(poll_interval)
            ready = self._enough_buffered(target_per_chunk)
            if not ready:
                continue
            self._emit_chunk(target_per_chunk)

    def _enough_buffered(self, n: int) -> bool:
        avail = []
        if self._mic is not None:
            avail.append(self._mic.buffered_samples())
        if self._bh is not None:
            avail.append(self._bh.buffered_samples())
        return all(a >= n for a in avail)

    def _emit_chunk(self, n: int) -> None:
        mic_audio = self._mic.drain(n) if self._mic is not None else None
        bh_audio = self._bh.drain(n) if self._bh is not None else None
        mixed = self._mix(mic_audio, bh_audio)
        if mixed.size == 0:
            return
        audio_16k = self._resample_to_whisper(mixed)
        chunk = TranscribeChunk(
            chunk_id=self._chunk_id,
            start_wall=datetime.now(),
            audio=audio_16k,
        )
        self._chunk_id += 1
        try:
            self.out_queue.put_nowait(chunk)
        except queue.Full:
            if self.on_overflow:
                self.on_overflow()

    def _flush_remaining(self) -> None:
        # Emit one final chunk from whatever's buffered at stop time.
        avail = []
        if self._mic is not None:
            avail.append(self._mic.buffered_samples())
        if self._bh is not None:
            avail.append(self._bh.buffered_samples())
        if not avail:
            return
        n = min(avail)
        if n < int(0.5 * self.capture_rate):
            return  # too short to be worth transcribing
        self._emit_chunk(n)

    def _mix(self, mic: np.ndarray | None, bh: np.ndarray | None) -> np.ndarray:
        if mic is not None and bh is not None:
            n = min(len(mic), len(bh))
            if n == 0:
                return np.zeros(0, dtype=np.float32)
            mixed = (mic[:n] + bh[:n]) * 0.5
            return np.clip(mixed, -1.0, 1.0).astype(np.float32, copy=False)
        if mic is not None:
            return mic
        if bh is not None:
            return bh
        return np.zeros(0, dtype=np.float32)

    def _resample_to_whisper(self, audio: np.ndarray) -> np.ndarray:
        if self.capture_rate == WHISPER_RATE:
            return audio.astype(np.float32, copy=False)
        # 48k -> 16k is up=1, down=3. For other rates, fall back to gcd reduction.
        from math import gcd
        g = gcd(self.capture_rate, WHISPER_RATE)
        up = WHISPER_RATE // g
        down = self.capture_rate // g
        out = resample_poly(audio, up, down)
        return out.astype(np.float32, copy=False)
