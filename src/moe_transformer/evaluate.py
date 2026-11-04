from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import _construct
from .data import TokenWindows, sha256_file, validate_processed_data
from .model import build_model
from .runtime import autocast_context, load_checkpoint, select_device, write_json


@torch.no_grad()
def evaluate(model, windows, batch_size, device, precision="fp32"):
    was_training = model.training
    model.eval()
    nll, count = 0.0, 0
    counts = {}
    try:
        for start in range(0, len(windows), batch_size):
            x, y = windows.get(list(range(start, min(start + batch_size, len(windows)))))
            with autocast_context(device, precision):
                logits, auxiliary = model(x.to(device), collect_router_stats=True)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                       y.to(device).reshape(-1), reduction="sum")
            nll += float(loss)
            count += y.numel()
            # Future MoE stats contract: one detached count tensor per layer,
            # or a dict containing an assignment_counts tensor.
            for layer, stats in auxiliary["routing_stats"].items():
                tensor = stats["assignment_counts"] if isinstance(stats, dict) else stats
                tensor = tensor.detach().cpu().long()
                counts[layer] = counts.get(layer, torch.zeros_like(tensor)) + tensor
    finally:
        model.train(was_training)
    ce = nll / count
    if not math.isfinite(ce):
        raise FloatingPointError("non-finite evaluation CE")
    return {"ce": ce, "perplexity": math.exp(ce), "target_tokens": count,
            "routing_counts": {k: v.tolist() for k, v in counts.items()}}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on development data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--output")
    args = parser.parse_args()
    saved = load_checkpoint(args.checkpoint)
    config = _construct(saved["config"])
    directory = Path(args.data_dir or config.data.output_dir)
    validate_processed_data(directory)
    if sha256_file(directory / "metadata.json") != saved["data_identity"]:
        raise ValueError("data/tokenizer identity differs from the checkpoint")
    device = select_device(args.device)
    model = build_model(config).to(device)
    model.load_state_dict(saved["model"])
    precision = args.precision or config.training.precision
    result = evaluate(model, TokenWindows(directory / "dev.npy", config.data.context_length),
                      config.training.micro_batch_size, device, precision)
    result.update(step=saved["step"], checkpoint=str(Path(args.checkpoint).resolve()),
                  precision=precision, device=str(device), split="dev")
    if args.output:
        write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
