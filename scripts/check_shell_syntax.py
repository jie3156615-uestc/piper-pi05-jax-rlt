from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bash -n over every shell script in a repository tree.")
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    scripts = sorted(args.root.resolve().rglob("*.sh"))
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
    print(f"bash syntax OK: {len(scripts)} scripts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
