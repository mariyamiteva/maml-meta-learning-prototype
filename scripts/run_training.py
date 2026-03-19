from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import train
from src.utils import load_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAML meta-training.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.json",
        help="Path to JSON config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    set_seed(config["seed"])
    train(config)


if __name__ == "__main__":
    main()
