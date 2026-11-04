from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset_id: str = "roneneldan/TinyStories"
    revision: str = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
    output_dir: str = "data/processed/tinystories_10m_bpe4096"
    target_train_tokens: int = 10_000_000
    target_dev_tokens: int = 200_000
    target_test_tokens: int = 200_000
    vocab_size: int = 4096
    context_length: int = 128


@dataclass
class ModelConfig:
    model_type: str = "dense"
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 768
    dropout: float = 0.1
    moe_layers: list[int] = field(default_factory=lambda: [1, 3])
    num_experts: int = 4
    top_k: int = 1


@dataclass
class MoEConfig:
    balance_policy: str = "fixed"
    alpha_start: float = 0.01
    alpha_end: float = 0.001
    z_loss_coefficient: float = 0.001


@dataclass
class TrainingConfig:
    max_steps: int = 5000
    micro_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    peak_learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    gradient_clip_norm: float = 1.0
    eval_interval: int = 500
    log_interval: int = 10
    precision: str = "fp16"
    output_dir: str = "runs"


@dataclass
class ExperimentConfig:
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if self.model.model_type not in {"dense", "moe"}:
            raise ValueError("model.model_type must be 'dense' or 'moe'")
        if self.model.d_model % self.model.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 1 <= self.model.top_k <= self.model.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        if any(i < 0 or i >= self.model.n_layers for i in self.model.moe_layers):
            raise ValueError("moe_layers contains an out-of-range zero-based layer index")
        if self.moe.balance_policy not in {"fixed", "scheduled", "zero"}:
            raise ValueError("moe.balance_policy must be fixed, scheduled, or zero")
        if self.training.micro_batch_size < 1 or self.training.gradient_accumulation_steps < 1:
            raise ValueError("batch and accumulation sizes must be positive")
        if self.training.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("training.precision must be fp32, fp16, or bf16")
        positive = (self.data.context_length, self.data.vocab_size, self.model.d_model,
                    self.model.n_heads, self.model.n_layers, self.model.d_ff,
                    self.training.max_steps, self.training.eval_interval, self.training.log_interval)
        if any(value <= 0 for value in positive):
            raise ValueError("dimensions, steps and intervals must be positive")
        if len(set(self.model.moe_layers)) != len(self.model.moe_layers):
            raise ValueError("moe_layers must be unique")
        if not 0 <= self.model.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 <= self.training.warmup_steps < self.training.max_steps:
            raise ValueError("warmup_steps must be below max_steps")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(data: dict[str, Any]) -> ExperimentConfig:
    config = ExperimentConfig(
        seed=int(data.get("seed", 0)),
        data=DataConfig(**data.get("data", {})),
        model=ModelConfig(**data.get("model", {})),
        moe=MoEConfig(**data.get("moe", {})),
        training=TrainingConfig(**data.get("training", {})),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return _construct(raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config.to_dict(), stream, sort_keys=False)

