"""List metrics for a specified wandb run."""

import argparse
import os
from pathlib import Path
import dotenv
import wandb


def main():
    dotenv.load_dotenv(Path(__file__).parent.parent / ".env")

    parser = argparse.ArgumentParser(description="List metrics for a wandb run")
    parser.add_argument("run_id", help="Run ID to inspect")
    parser.add_argument(
        "--entity", default=None, help="Override WANDB_ENTITY from .env"
    )
    parser.add_argument(
        "--project", default=None, help="Override WANDB_PROJECT from .env"
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show metric history instead of summary",
    )
    parser.add_argument(
        "--keys",
        default=None,
        help="Comma-separated list of metric keys to show (default: all)",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=10,
        metavar="N",
        help="When showing history, show last N rows (default: 10)",
    )
    args = parser.parse_args()

    api = wandb.Api()
    entity = args.entity or os.environ.get("WANDB_ENTITY") or api.viewer.username
    project = args.project or os.environ.get("WANDB_PROJECT")

    if not project:
        parser.error("project is required: set WANDB_PROJECT in .env or pass --project")

    try:
        run = api.run(f"{entity}/{project}/{args.run_id}")
    except Exception as e:
        print(f"Error fetching run: {e}")
        return

    print(f"Run:     {run.name} ({run.id})")
    print(f"State:   {run.state}")
    print(f"Project: {entity}/{project}")
    print()

    filter_keys = set(args.keys.split(",")) if args.keys else None

    if args.history:
        history = run.history()
        if filter_keys:
            cols = [c for c in history.columns if c in filter_keys or c == "_step"]
        else:
            cols = list(history.columns)

        df = history[cols].tail(args.last)
        print(df.to_string(index=False))
    else:
        summary = {k: v for k, v in run.summary.items() if k.startswith("eval/")}
        if filter_keys:
            summary = {k: v for k, v in summary.items() if k in filter_keys}

        if not summary:
            print("No metrics found in summary.")
            return

        key_width = max(len(k) for k in summary)
        print(f"{'Metric':<{key_width}}  Value")
        print("-" * (key_width + 20))
        for key, value in sorted(summary.items()):
            if isinstance(value, float):
                print(f"{key:<{key_width}}  {value:.6g}")
            else:
                print(f"{key:<{key_width}}  {value}")


if __name__ == "__main__":
    main()
