from __future__ import annotations

import argparse
from pathlib import Path

from piper_runtime.rlt_token_runtime import export_encoder_only_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip decoder weights from a JAX RL-token checkpoint")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_encoder_only_checkpoint(args.source, args.output)
    print(result)


if __name__ == "__main__":
    main()
