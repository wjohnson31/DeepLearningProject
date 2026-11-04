"""Small correctness checks, optionally using the real prepared training data."""
import argparse
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import load_config
from .data import DeterministicBatcher, TokenWindows, validate_processed_data
from .model import build_model
from .runtime import atomic_checkpoint, load_checkpoint, seed_all, select_device, write_json


def run_checks(config, device, updates=20):
    metadata = validate_processed_data(config.data.output_dir)
    seed_all(config.seed)
    windows = TokenWindows(Path(config.data.output_dir) / "train.npy", config.data.context_length)
    batcher = DeterministicBatcher(windows, min(2, len(windows)), config.seed)
    x, y = batcher.next()
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert int(x.min()) >= 0 and int(x.max()) < config.data.vocab_size
    saved_batcher = batcher.state_dict()
    next_x, _ = batcher.next()
    batcher.load_state_dict(saved_batcher)
    assert torch.equal(next_x, batcher.next()[0])
    x, y = x.to(device), y.to(device)
    model = build_model(config).to(device)
    model.eval()
    with torch.no_grad():
        logits, aux = model(x)
        assert list(logits.shape) == [len(x), config.data.context_length, config.data.vocab_size]
        assert aux["balance_loss"].item() == 0 and aux["router_z_loss"].item() == 0
        assert aux["routing_stats"] == {}
        altered = x.clone()
        split = x.shape[1] // 2
        altered[:, split:] = (altered[:, split:] + 1) % config.data.vocab_size
        changed_logits, _ = model(altered)
        causal_error = float((logits[:, :split] - changed_logits[:, :split]).abs().max())
        torch.testing.assert_close(logits[:, :split], changed_logits[:, :split], rtol=0, atol=1e-6)
        initial_ce = float(F.cross_entropy(logits.reshape(-1, config.data.vocab_size), y.reshape(-1)))
    initial_weights = model.token_embedding.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    # Disable dropout deliberately to make the repeated-batch check stable.
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, config.data.vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    with torch.no_grad():
        logits, _ = model(x)
        final_ce = float(F.cross_entropy(logits.reshape(-1, config.data.vocab_size), y.reshape(-1)))
    assert final_ce < initial_ce
    assert not torch.equal(initial_weights, model.token_embedding.weight)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "check.pt"
        atomic_checkpoint(path, {"model": model.state_dict()})
        restored = build_model(config).to(device)
        restored.load_state_dict(load_checkpoint(path)["model"])
        restored.eval()
        with torch.no_grad():
            reloaded, _ = restored(x)
        torch.testing.assert_close(logits, reloaded, rtol=0, atol=0)
    return {
        "status": "passed", "source": metadata["source"], "device": str(device),
        "logit_shape": list(logits.shape), "parameters": model.unique_parameter_count(),
        "causal_max_absolute_error": causal_error, "repeated_batch_updates": updates,
        "repeated_batch_initial_ce": initial_ce, "repeated_batch_final_ce": final_ce,
        "checkpoint_predictions_identical": True, "batch_shift_and_resume_passed": True,
        "interpretation": "correctness check only; not a trained baseline result",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/milestone.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="artifacts/checks.json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = run_checks(load_config(args.config), select_device(args.device))
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
