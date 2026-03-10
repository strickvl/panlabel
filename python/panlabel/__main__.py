from __future__ import annotations

import os
import sys

from panlabel import find_panlabel_bin


def _run() -> None:
    panlabel = find_panlabel_bin()

    if sys.platform == "win32":
        import subprocess

        try:
            completed_process = subprocess.run([panlabel, *sys.argv[1:]])
        except KeyboardInterrupt:
            sys.exit(2)
        sys.exit(completed_process.returncode)
    else:
        os.execvp(panlabel, [panlabel, *sys.argv[1:]])


if __name__ == "__main__":
    _run()
