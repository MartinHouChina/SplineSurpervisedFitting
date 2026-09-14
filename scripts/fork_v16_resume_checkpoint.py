"""Fork a v16 .last.pt so validation can change without losing learned heads."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-last", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--synthetic-boundary-val-size", type=int, required=True)
    args = parser.parse_args()

    source_last = args.source_last.resolve()
    output = args.output.resolve()
    if not source_last.name.endswith(".last.pt"):
        parser.error("--source-last must end in .last.pt")
    if output.suffix != ".pt" or output.name.endswith(".last.pt"):
        parser.error("--output must be the new best-checkpoint .pt path")
    if args.synthetic_boundary_val_size < 32:
        parser.error("formal fork requires at least 32 K=56 boundary samples")

    new_last = output.with_name(output.stem + ".last.pt")
    new_proposal = output.with_name(output.stem + ".proposal.pt")
    new_history = output.with_suffix(".history.json")
    targets = (output, new_last, new_proposal, new_history)
    existing = [path for path in targets if path.exists()]
    if existing:
        parser.error("refusing to overwrite: " + ", ".join(map(str, existing)))

    payload = torch.load(source_last, map_location="cpu", weights_only=True)
    source_epoch = int(payload["epoch"])
    if args.epochs != source_epoch + 5:
        parser.error(
            f"requested total epochs must equal source epoch + 5 "
            f"({source_epoch + 5})"
        )
    config = dict(payload["training_config"])
    config.update(
        epochs=args.epochs,
        output=str(output),
        synthetic_boundary_val_size=args.synthetic_boundary_val_size,
        resume=None,
        init_checkpoint=config.get("init_checkpoint"),
    )
    payload["training_config"] = config
    # Validation changes invalidate the old best rank. The first new epoch
    # must establish the best checkpoint under the new validation evidence.
    payload["best_joint_rank"] = None

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, new_last)
    # train_v16 requires a best artifact to exist for a joint-stage resume;
    # it will be overwritten after the first newly validated epoch.
    torch.save(payload, output)

    source_base = source_last.name[: -len(".last.pt")]
    source_proposal = source_last.with_name(source_base + ".proposal.pt")
    if not source_proposal.is_file():
        parser.error(f"source proposal artifact is missing: {source_proposal}")
    shutil.copy2(source_proposal, new_proposal)
    print(f"Forked full epoch-{source_epoch} state to {new_last}")


if __name__ == "__main__":
    main()
