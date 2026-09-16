#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
Build helper for the PolyCycle VCL Toolkit.

Usage
-----
    python build.py            # normal build
    python build.py --clean    # wipe build/ and dist/ first
    python build.py --onefile  # build a single-file executable (larger, slower startup)

The result is placed in  dist/vcl_toolkit/  (one-folder) or  dist/vcl_toolkit  (one-file).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SPEC = _ROOT / "vcl_toolkit.spec"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VCL Toolkit binary")
    parser.add_argument("--clean", action="store_true",
                        help="Remove build/ and dist/ before building")
    parser.add_argument("--onefile", action="store_true",
                        help="Produce a single-file executable instead of a folder")
    args = parser.parse_args()

    if args.clean:
        for d in ("build", "dist"):
            p = _ROOT / d
            if p.exists():
                print(f"Removing {p} …")
                shutil.rmtree(p)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        str(_SPEC),
    ]

    if args.onefile:
        # Override the COLLECT step with --onefile.
        # Note: one-file mode re-extracts everything to a temp dir at startup,
        # so first launch is slower.  Prefer one-folder for dev / fast start.
        cmd.append("--onefile")

    print(f"▶ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(_ROOT))

    if result.returncode == 0:
        print("\n✅  Build succeeded!")
        if args.onefile:
            print(f"   Binary: {_ROOT / 'dist' / 'vcl_toolkit'}")
        else:
            print(f"   Folder: {_ROOT / 'dist' / 'vcl_toolkit' / ''}")
        print("\n   To run:  ./dist/vcl_toolkit/vcl_toolkit")
        print("   (Set VCL_PYTHON=/path/to/python3 if the target machine's")
        print("    Python is not on PATH.)")
    else:
        print(f"\n❌  Build failed (exit code {result.returncode})")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
