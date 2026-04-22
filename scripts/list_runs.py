"""List all wandb runs for a given project."""

import argparse
import os
from pathlib import Path
import dotenv
import wandb


def main():
    dotenv.load_dotenv(Path(__file__).parent.parent / ".env")

    parser = argparse.ArgumentParser(description="List wandb runs")
    parser.add_argument(
        "--entity", default=None, help="Override WANDB_ENTITY from .env"
    )
    parser.add_argument(
        "--project", default=None, help="Override WANDB_PROJECT from .env"
    )
    parser.add_argument(
        "--state",
        default=None,
        help="Filter by state: running, finished, crashed, failed",
    )
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    api = wandb.Api()
    entity = args.entity or os.environ.get("WANDB_ENTITY") or api.viewer.username
    project = args.project or os.environ.get("WANDB_PROJECT")

    if not project:
        parser.error("project is required: set WANDB_PROJECT in .env or pass --project")

    filters = {}
    if args.state:
        filters["state"] = args.state

    try:
        runs = api.runs(f"{entity}/{project}", filters=filters, order="-created_at")
        print(f"{'ID':<12} {'State':<10} {'Name':<40} {'Created':<20} {'Steps':>8}")
        print("-" * 96)
        count = 0
        for run in runs:
            if count >= args.limit:
                break
            created = run.created_at[:19].replace("T", " ") if run.created_at else "-"
            steps = run.summary.get("_step", "-")
            print(
                f"{run.id:<12} {run.state:<10} {run.name[:40]:<40} {created:<20} {str(steps):>8}"
            )
            count += 1
        if count == 0:
            print(f"No runs found in '{entity}/{project}'")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
