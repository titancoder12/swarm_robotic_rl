from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["custom", "sb3", "rllib"],
        default="custom",
        help="Training backend (default: custom)",
    )
    args, remaining = parser.parse_known_args()

    if args.backend == "custom":
        from train.independent_dqn_pytorch import parse_args, train

        train(parse_args(remaining))
        return

    if args.backend == "sb3":
        from train.sb3_dqn import parse_args, run

        run(parse_args(remaining))
        return

    if args.backend == "rllib":
        from train.rllib_dqn import parse_args, run

        run(parse_args(remaining))
        return

    raise ValueError(f"Unsupported backend: {args.backend}")


if __name__ == "__main__":
    main()
