from __future__ import annotations

import shutil
import subprocess


def copy_to_clipboard(text: str) -> bool:
    if shutil.which("pbcopy") is None:
        return False
    try:
        subprocess.run(["pbcopy"], input=text, text=True, check=True)
        return True
    except subprocess.SubprocessError:
        return False
