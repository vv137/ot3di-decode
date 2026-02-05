"""CLI entry point."""

import argparse
from pathlib import Path

from .train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OT3Di model")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML path")
    parser.add_argument("--data", type=Path, required=True, help="Training data JSON path")
    parser.add_argument("--output", type=Path, default="./output", help="Output directory")

    args = parser.parse_args()
    train(args.config, args.data, args.output)


if __name__ == "__main__":
    main()
