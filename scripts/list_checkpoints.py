"""List and optionally download checkpoints for a given wandb run."""

import argparse
import os
from pathlib import Path
import dotenv
import wandb


def fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def main():
    dotenv.load_dotenv(Path(__file__).parent.parent / ".env")

    parser = argparse.ArgumentParser(
        description="List and download checkpoints for a wandb run"
    )
    parser.add_argument("run_id", help="Run ID (e.g. x6zyq2o1)")
    parser.add_argument(
        "--entity", default=None, help="Override WANDB_ENTITY from .env"
    )
    parser.add_argument(
        "--project", default=None, help="Override WANDB_PROJECT from .env"
    )
    parser.add_argument(
        "--download",
        metavar="VERSION",
        help="Download a specific version, e.g. v2. Use 'latest' for the latest.",
    )
    parser.add_argument(
        "--output-dir", default=".", help="Directory to download into (default: .)"
    )
    args = parser.parse_args()

    api = wandb.Api()
    entity = args.entity or os.environ.get("WANDB_ENTITY") or api.viewer.username
    project = args.project or os.environ.get("WANDB_PROJECT")

    if not project:
        parser.error("project is required: set WANDB_PROJECT in .env or pass --project")

    run = api.run(f"{entity}/{project}/{args.run_id}")
    artifacts = sorted(
        [a for a in run.logged_artifacts() if a.type == "model"],
        key=lambda x: x.metadata.get("global_step", 0),
    )

    if not artifacts:
        print(f"No checkpoints found for run '{args.run_id}'")
        return

    if args.download:
        version = "latest" if args.download == "latest" else args.download
        name = f"checkpoint-{args.run_id}:{version}"
        artifact = api.artifact(f"{entity}/{project}/{name}", type="model")
        out = Path(args.output_dir)
        print(f"Downloading {artifact.name} ({fmt_size(artifact.size)}) -> {out}/")
        artifact.download(root=str(out))
        print("Done.")
        return

    print(f"Run: {run.name} ({args.run_id})  state={run.state}")
    print()
    print(f"{'Version':<28} {'Step':>8} {'Size':>10}  {'Created'}")
    print("-" * 72)
    for a in artifacts:
        step = a.metadata.get("global_step", "-")
        created = a.created_at[:19].replace("T", " ") if a.created_at else "-"
        print(f"{a.name:<28} {str(step):>8} {fmt_size(a.size):>10}  {created}")


if __name__ == "__main__":
    main()
