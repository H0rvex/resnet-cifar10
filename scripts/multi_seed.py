"""Run training for several seeds and write mean ± std of best test accuracy."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import statistics
from pathlib import Path

from resnet_cifar10.train import build_parser, resolve_config, train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train with multiple seeds using the same YAML recipe; aggregate results."
    )
    p.add_argument(
        "--config", required=True, metavar="PATH", help="Base YAML config (same as train.py)"
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[41, 42, 43],
        metavar="N",
        help="Seeds to run (default: 41 42 43)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        metavar="PATH",
        help="Directory for summary.json (default: runs/multi_seed_<timestamp>)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join("runs", f"multi_seed_{ts}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    base_parser = build_parser()
    runs: list[dict] = []
    for seed in args.seeds:
        ns = base_parser.parse_args(["--config", args.config, "--seed", str(seed)])
        cfg = resolve_config(ns)
        print(f"\n=== Training seed={seed} ===")
        result = train(cfg)
        runs.append(
            {
                "seed": seed,
                "best_acc": round(result.best_acc, 4),
                "wall_time_sec": round(result.wall_time_sec, 2),
                "run_dir": result.run_dir,
                "best_checkpoint": result.best_checkpoint,
            }
        )

    accs = [r["best_acc"] for r in runs]
    summary = {
        "config_path": os.path.abspath(args.config),
        "seeds": list(args.seeds),
        "mean_best_acc": round(statistics.mean(accs), 4),
        "stdev_best_acc": round(statistics.stdev(accs), 4) if len(accs) > 1 else 0.0,
        "runs": runs,
    }
    out_path = os.path.join(out_dir, "summary.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(
        f"\nAggregated {len(runs)} runs → mean {summary['mean_best_acc']:.2f}% ± {summary['stdev_best_acc']:.2f}%"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
