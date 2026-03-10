from __future__ import annotations

import os
import sys
import sysconfig


def find_panlabel_bin() -> str:
    """Return the panlabel binary path."""
    panlabel_exe = "panlabel" + sysconfig.get_config_var("EXE")

    scripts_dirs = [
        sysconfig.get_path("scripts"),
        sysconfig.get_path("scripts", vars={"base": sys.base_prefix}),
    ]

    for scripts_dir in scripts_dirs:
        if scripts_dir:
            path = os.path.join(scripts_dir, panlabel_exe)
            if os.path.isfile(path):
                return path

    raise FileNotFoundError(
        "Could not find the panlabel binary. Is panlabel installed?"
    )
