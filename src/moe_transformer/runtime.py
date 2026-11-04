from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def select_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a GPU runtime or pass --device cpu for checks")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process")
    return device


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("Mixed precision is supported here only on CUDA; use --precision fp32")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; use fp16")
    return torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def rng_state():
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    name, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def source_identity():
    root = Path(__file__).parent
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                         text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return {"sha256": digest, "files": files, "git_commit": commit}


def environment(device, precision):
    versions = {}
    for name in ("torch", "numpy", "tokenizers", "requests", "PyYAML", "matplotlib"):
        versions[name] = importlib.metadata.version(name)
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "device": str(device), "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda" else platform.machine(),
        "precision": precision, "versions": versions,
    }


def atomic_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def forward_flops(config):
    """Analytical dense matmul estimate; 2 FLOPs per multiply-add.

    Includes QKV/output projections, full square QK/AV attention, FFNs and
    vocabulary head. Excludes scalar activations/norm/softmax/embedding lookup.
    This is an explicit arithmetic convention, not measured hardware FLOPs.
    Person 2 must extend this formula to include routing and active experts.
    """
    if config.model.model_type != "dense":
        raise ValueError("Add the MoE/router FLOP formula before reporting MoE costs")
    m, s, v = config.model, config.data.context_length, config.data.vocab_size
    per_block = 8 * m.d_model**2 + 4 * s * m.d_model + 4 * m.d_model * m.d_ff
    return m.n_layers * per_block + 2 * m.d_model * v
