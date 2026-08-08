#!/usr/bin/env python3
"""Print selected top-level JSON fields, one value per line, for Bash hooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--field", action="append", required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    for field in args.field:
        value = payload.get(field, "")
        if isinstance(value, (dict, list)):
            print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        elif value is None:
            print("")
        else:
            print(value)


if __name__ == "__main__":
    main()
