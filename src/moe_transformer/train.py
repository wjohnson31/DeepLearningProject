from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import load_config, save_config
from .data import DeterministicBatcher, TokenWindows, sha256_file, validate_processed_data
from .evaluate import evaluate
from .model import build_model
from .runtime import (atomic_checkpoint, autocast_context, environment, forward_flops,
                      load_checkpoint, restore_rng, rng_state, seed_all, select_device,
                      source_identity, synchronize, write_json)
from .schedules import balancing_coefficient, cosine_learning_rate


def make_optimizer(model, config):
    # All 1D parameters are biases or normalization scale/bias in this decoder.
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    t = config.training
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": t.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=t.peak_learning_rate, betas=(t.beta1, t.beta2), eps=t.epsilon,
    )


def comparable_config(config):
    data = config.to_dict()
    data["data"].pop("output_dir")
    data["training"].pop("output_dir")
    return data


def training_update(model, optimizer, scaler, batcher, config, device, alpha):
    """Retry FP16 overflow without advancing the data order or update schedule."""
    t = config.training
    original_batch = batcher.state_dict()
    original_rng = rng_state() if scaler.is_enabled() else None
    for attempt in range(10):
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(4, device=device)
        for _ in range(t.gradient_accumulation_steps):
            x, y = batcher.next()
            with autocast_context(device, t.precision):
                logits, aux = model(x.to(device))
                ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.to(device).reshape(-1))
                balance, z_loss = aux["balance_loss"], aux["router_z_loss"]
                total = ce + alpha * balance + config.moe.z_loss_coefficient * z_loss
            scaler.scale(total / t.gradient_accumulation_steps).backward()
            totals += torch.stack((ce.detach(), balance.detach(), z_loss.detach(), total.detach()))
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.gradient_clip_norm,
                                                   error_if_nonfinite=False)
        if not torch.isfinite(grad_norm):
            if not scaler.is_enabled():
                raise FloatingPointError("Nonfinite FP32 gradients; inspect the saved run")
            scaler.update(new_scale=scaler.get_scale() / 2)
            batcher.load_state_dict(original_batch)
            restore_rng(original_rng)
            continue
        scaler.step(optimizer)
        scaler.update()
        return totals, grad_norm, attempt
    raise FloatingPointError("FP16 gradients remained nonfinite after 10 reduced-scale attempts")


def train(config, run_dir, device, stop_after=None, resume=None, profile_warmup=20):
    config.validate()
    # Fail early for unsupported device/precision combinations.
    with autocast_context(device, config.training.precision):
        pass
    data_dir = Path(config.data.output_dir)
    metadata = validate_processed_data(data_dir)
    if metadata["vocab_size"] != config.data.vocab_size or metadata["context_length"] != config.data.context_length:
        raise ValueError("config vocabulary/context do not match prepared data")
    identity = sha256_file(data_dir / "metadata.json")
    t = config.training
    end = t.max_steps if stop_after is None else min(stop_after, t.max_steps)
    checkpoint = load_checkpoint(resume) if resume else None
    if checkpoint and checkpoint.get("checkpoint_kind") == "evaluation":
        raise ValueError("best.pt is evaluation-only; resume training from last.pt")
    start = checkpoint["step"] if checkpoint else 0
    if end <= start:
        raise ValueError("stop-after must exceed the checkpoint's completed step")
    if checkpoint:
        from .config import _construct
        if comparable_config(config) != comparable_config(_construct(checkpoint["config"])):
            raise ValueError("resume configuration differs; preserve the planned horizon and training protocol")
        if identity != checkpoint["data_identity"]:
            raise ValueError("resume data/tokenizer identity differs")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    save_config(config, run_dir / "config.yaml")
    env = environment(device, t.precision)
    code = source_identity()
    write_json(run_dir / "environment.json", env)
    write_json(run_dir / "code_identity.json", code)
    shutil.copytree(Path(__file__).parent, run_dir / "source", ignore=shutil.ignore_patterns("__pycache__"))
    seed_all(config.seed)
    model = build_model(config).to(device)
    optimizer = make_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and t.precision == "fp16")
    batcher = DeterministicBatcher(TokenWindows(data_dir / "train.npy", config.data.context_length),
                                   t.micro_batch_size, config.seed)
    dev = TokenWindows(data_dir / "dev.npy", config.data.context_length)
    history = []
    best_ce, best_step = float("inf"), 0
    best_snapshot = None
    training_seconds = elapsed_seconds = profile_seconds = 0.0
    profile_tokens = 0
    peak_memory = 0 if device.type == "cuda" else None
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        batcher.load_state_dict(checkpoint["batcher"])
        restore_rng(checkpoint["rng"])
        history = checkpoint["evaluations"]
        best_ce, best_step = checkpoint["best_ce"], checkpoint["best_step"]
        training_seconds = checkpoint["training_seconds"]
        elapsed_seconds = checkpoint["elapsed_seconds"]
        # Profiling is session/device-specific; a new session re-warms up.
        best_snapshot = checkpoint.get("best_snapshot")
        source_best = Path(resume).parent / "best.pt"
        if best_snapshot is not None:
            # The last checkpoint is authoritative even if a disconnect left the
            # separate best file missing, older, or ahead of the last checkpoint.
            atomic_checkpoint(run_dir / "best.pt", best_snapshot)
        elif source_best.exists():
            prior_best = load_checkpoint(source_best)
            if (prior_best["step"] != best_step or prior_best["data_identity"] != identity
                    or prior_best["best_ce"] != best_ce):
                raise ValueError("legacy parent best checkpoint does not match resume checkpoint")
            best_snapshot = {key: prior_best[key] for key in (
                "model", "config", "data_identity", "code_identity", "environment",
                "step", "best_ce", "best_step")}
            best_snapshot["checkpoint_kind"] = "evaluation"
            atomic_checkpoint(run_dir / "best.pt", best_snapshot)
        else:
            raise FileNotFoundError("legacy checkpoint requires best.pt beside last.pt")
    else:
        initial = evaluate(model, dev, t.micro_batch_size, device, t.precision)
        history.append({"step": 0, "token_exposures": 0, **initial})
    tokens_per_update = t.micro_batch_size * t.gradient_accumulation_steps * config.data.context_length
    session_started = time.perf_counter()
    previous_elapsed = elapsed_seconds
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with (run_dir / "train.jsonl").open("w") as log:
        for step in range(start, end):
            synchronize(device)
            update_started = time.perf_counter()
            lr = cosine_learning_rate(step, t.max_steps, t.warmup_steps,
                                      t.peak_learning_rate, t.min_learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = lr
            alpha = balancing_coefficient(step, t.max_steps, config.moe.balance_policy,
                                          config.moe.alpha_start, config.moe.alpha_end)
            totals, grad_norm, retries = training_update(model, optimizer, scaler, batcher, config, device, alpha)
            synchronize(device)
            seconds = time.perf_counter() - update_started
            training_seconds += seconds
            if step - start >= profile_warmup:
                profile_seconds += seconds
                profile_tokens += tokens_per_update
            if device.type == "cuda":
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
            values = (totals / t.gradient_accumulation_steps).cpu().tolist()
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError("non-finite training losses")
            completed = step + 1
            row = dict(step=completed, token_exposures=completed * tokens_per_update,
                       ce=values[0], balance_loss=values[1], router_z_loss=values[2],
                       total_loss=values[3], alpha=alpha if config.model.model_type == "moe" else 0.0,
                       learning_rate=lr, gradient_norm=float(grad_norm), update_seconds=seconds,
                       training_seconds=training_seconds, fp16_overflow_retries=retries)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if completed % t.log_interval == 0 or completed == end:
                print(f"step {completed}/{t.max_steps} CE={values[0]:.4f} "
                      f"tokens={row['token_exposures']:,} update={seconds:.3f}s", flush=True)
            if completed % t.eval_interval == 0 or completed == end:
                result = evaluate(model, dev, t.micro_batch_size, device, t.precision)
                history.append({"step": completed, "token_exposures": completed * tokens_per_update, **result})
                improved = result["ce"] < best_ce
                if improved:
                    best_ce, best_step = result["ce"], completed
                    best_snapshot = dict(
                        checkpoint_kind="evaluation", step=completed,
                        model={name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                        config=config.to_dict(), data_identity=identity, code_identity=code,
                        environment=env, best_ce=best_ce, best_step=best_step,
                    )
                elapsed_seconds = previous_elapsed + time.perf_counter() - session_started
                saved = dict(
                    schema_version=2, checkpoint_kind="training", best_snapshot=best_snapshot,
                    config=config.to_dict(), data_identity=identity,
                    code_identity=code, environment=env, step=completed,
                    model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(), rng=rng_state(), batcher=batcher.state_dict(),
                    scheduler={"completed_steps": completed, "planned_horizon": t.max_steps},
                    best_ce=best_ce, best_step=best_step, evaluations=history,
                    training_seconds=training_seconds, elapsed_seconds=elapsed_seconds,
                )
                if improved:
                    atomic_checkpoint(run_dir / "best.pt", best_snapshot)
                atomic_checkpoint(run_dir / "last.pt", saved)
                write_json(run_dir / "evaluations.json", history)
                throughput = profile_tokens / profile_seconds if profile_seconds else None
                summary = dict(
                    status="complete" if completed == t.max_steps else "partial",
                    source=metadata["source"], seed=config.seed, model_type=config.model.model_type,
                    parameters=model.unique_parameter_count(), planned_steps=t.max_steps,
                    completed_steps=completed, token_exposures=completed * tokens_per_update,
                    selected_checkpoint_step=best_step, dev_ce=best_ce, dev_perplexity=math.exp(best_ce),
                    final_dev_ce=result["ce"], final_dev_perplexity=result["perplexity"],
                    forward_matmul_flops_per_token=forward_flops(config),
                    training_seconds=training_seconds, elapsed_seconds=elapsed_seconds,
                    session_profile_tokens_per_second=throughput,
                    session_profile_seconds=profile_seconds, session_profile_tokens=profile_tokens,
                    profile_warmup_updates=profile_warmup, peak_training_cuda_memory_bytes=peak_memory,
                    environment=env, data_identity=identity, code_identity=code["sha256"],
                    parent_checkpoint=str(Path(resume).resolve()) if resume else None,
                    run_directory=str(run_dir.resolve()),
                )
                write_json(run_dir / "results.json", summary)
                print(f"  dev CE={result['ce']:.5f} PPL={result['perplexity']:.3f}; "
                      f"best step={best_step}", flush=True)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train the dense D1 baseline or resume a saved run")
    parser.add_argument("--config", default="configs/milestone.yaml")
    parser.add_argument("--run-dir", required=True, help="New directory; use a new name on resume")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--stop-after", type=int, help="Stop at this absolute update; preserve planned horizon")
    parser.add_argument("--resume", help="Path to last.pt, with best.pt beside it")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    if args.precision:
        config.training.precision = args.precision
    print(json.dumps(train(config, args.run_dir, select_device(args.device),
                           args.stop_after, args.resume), indent=2))


if __name__ == "__main__":
    main()
