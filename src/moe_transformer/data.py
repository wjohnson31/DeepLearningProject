from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_token_array(path: str | Path, tokens: list[int] | np.ndarray) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(tokens, dtype=np.int32)
    np.save(destination, array, allow_pickle=False)
    return {
        "path": destination.name,
        "tokens": int(array.size),
        "sha256": sha256_file(destination),
    }


class TokenWindows:
    """Memory-mapped, fixed 129-token windows represented as 128 inputs/targets."""

    def __init__(self, path: str | Path, context_length: int) -> None:
        self.path = Path(path)
        self.tokens = np.load(self.path, mmap_mode="r", allow_pickle=False)
        self.context_length = context_length
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.tokens.ndim != 1:
            raise ValueError(f"{self.path} must contain a one-dimensional token array")
        self.window_count = max(0, (len(self.tokens) - 1) // context_length)
        if self.window_count == 0:
            raise ValueError(f"{self.path} is too short for context length {context_length}")

    def __len__(self) -> int:
        return self.window_count

    def get(self, indices: Tensor | list[int]) -> tuple[Tensor, Tensor]:
        if isinstance(indices, Tensor):
            selected = indices.detach().cpu().tolist()
        else:
            selected = indices
        if not selected or any(index < 0 or index >= len(self) for index in selected):
            raise ValueError("window indices must be nonempty and within range")
        windows = np.stack(
            [
                np.asarray(
                    self.tokens[index * self.context_length : index * self.context_length + self.context_length + 1],
                    dtype=np.int64,
                )
                for index in selected
            ]
        )
        batch = torch.from_numpy(windows)
        return batch[:, :-1], batch[:, 1:]


@dataclass
class BatcherState:
    epoch: int
    position: int


class DeterministicBatcher:
    """Deterministic shuffled batches with compact, checkpointable state."""

    def __init__(self, windows: TokenWindows, batch_size: int, seed: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if batch_size > len(windows):
            raise ValueError("training data must contain at least one full microbatch")
        self.windows = windows
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.position = 0
        self._order = self._make_order()

    def _make_order(self) -> Tensor:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.windows), generator=generator)

    def next(self) -> tuple[Tensor, Tensor]:
        if self.position + self.batch_size > len(self._order):
            self.epoch += 1
            self.position = 0
            self._order = self._make_order()
        indices = self._order[self.position : self.position + self.batch_size]
        self.position += self.batch_size
        return self.windows.get(indices)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "position": self.position, "seed": self.seed}

    def load_state_dict(self, state: dict[str, int]) -> None:
        if int(state["seed"]) != self.seed:
            raise ValueError("batcher seed does not match checkpoint")
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
        if not 0 <= self.position <= len(self.windows):
            raise ValueError("invalid batcher position")
        self._order = self._make_order()


def load_metadata(data_directory: str | Path) -> dict[str, Any]:
    path = Path(data_directory) / "metadata.json"
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_processed_data(data_directory: str | Path) -> dict[str, Any]:
    directory = Path(data_directory)
    metadata = load_metadata(directory)
    tokenizer = metadata.get("tokenizer")
    if tokenizer and sha256_file(directory / tokenizer["path"]) != tokenizer["sha256"]:
        raise ValueError("tokenizer checksum mismatch")
    for split in ("train", "dev", "test"):
        record = metadata["splits"][split]
        path = directory / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"checksum mismatch for {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if int(array.size) != int(record["tokens"]):
            raise ValueError(f"token count mismatch for {path}")
        if array.size and (int(array.min()) < 0 or int(array.max()) >= int(metadata["vocab_size"])):
            raise ValueError(f"token ID out of range in {path}")
    train_ids = set(metadata["story_ids"]["train"])
    dev_ids = set(metadata["story_ids"]["dev"])
    test_ids = set(metadata["story_ids"]["test"])
    if train_ids & dev_ids or train_ids & test_ids or dev_ids & test_ids:
        raise ValueError("story hashes overlap across splits")
    return metadata

