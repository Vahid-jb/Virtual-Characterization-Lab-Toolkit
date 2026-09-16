#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

import sys

from compute_saed import ( 
    ASFSAED_NP,
    ENGINES,
    NUMBA_AVAILABLE,
    PSUTIL_AVAILABLE,
    ComputeSAED,
    resolve_engine,
    _DEFAULT_CHUNK,
    _kernel_numba,
    _kernel_numpy,
    _kernel_reference,
)
import compute_saed


def main():
    argv = sys.argv[1:]
    if not any(a == "--engine" or a.startswith("--engine=") for a in argv):
        sys.argv = [sys.argv[0]] + argv + ["--engine", "numba"]
    compute_saed.main()


if __name__ == "__main__":
    main()
