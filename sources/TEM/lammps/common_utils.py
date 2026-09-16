#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
common_utils.py - Input parsing for compute_saed.py.

A copy of the parser in XRD/common_utils.py, so the SAED script runs on its own.
"""

import os

# Input file parsing

_TRUE_WORDS = ('yes', 'true', 'y', 'on')
_FALSE_WORDS = ('no', 'false', 'n', 'off')


def _convert_scalar(value):
    text = value.strip()
    if text == '':
        return ''
    low = text.lower()
    if low in _TRUE_WORDS:
        return True
    if low in _FALSE_WORDS:
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _split_list(value):
    text = value.strip()
    if len(text) >= 2 and text[0] in '([' and text[-1] in ')]':
        text = text[1:-1]
    parts = [p.strip() for p in text.split(',')]
    parts = [p for p in parts if p != '']
    return [_convert_scalar(p) for p in parts]


def parse_input_file(input_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    params = {}
    current_section = 'general'

    with open(input_file, 'r', encoding='utf-8-sig', errors='replace') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith('#') or line.startswith('!'):
                continue

            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1].strip().lower()
                continue

            if '=' not in line:
                continue

            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            # strip inline comments
            for marker in ('#', '!'):
                if marker in value:
                    value = value.split(marker, 1)[0].strip()

            # strip surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]

            bracketed = (len(value) >= 2 and value[0] in '([' and value[-1] in ')]')
            if (',' in value) or bracketed:
                converted = _split_list(value)
                if len(converted) == 1 and not bracketed:
                    converted = converted[0]
            else:
                converted = _convert_scalar(value)

            full_key = f"{current_section}.{key}" if current_section != 'general' else key
            params[full_key] = converted

    return params


def validate_params(params, required_keys, module_name):
    missing = [key for key in required_keys if key not in params]
    if missing:
        print(f"Error: missing required parameters for {module_name}:")
        for key in missing:
            print(f"  - {key}")
        return False
    return True
