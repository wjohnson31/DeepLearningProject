from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


Auxiliary = dict[str, Any]
FFNFactory = Callable[[int, ModelConfig], nn.Module]


class DenseFFN(nn.Module):
    """The dense FFN used by D1 and as the replaceable block interface."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.down(F.gelu(self.up(inputs)))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, sequence, width = inputs.shape
        qkv = self.qkv(inputs).view(batch, sequence, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, width)
        return self.residual_dropout(self.output(attended))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, ffn: nn.Module) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config.d_model, config.n_heads, config.dropout)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = ffn
        self.ffn_dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor, collect_router_stats: bool = False) -> tuple[Tensor, Auxiliary]:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        ffn_inputs = self.ffn_norm(hidden)

        # Person 2 can replace an FFN with a module that returns (output, aux).
        # DenseFFN returns only a tensor, which maps to zero/empty auxiliary data.
        result = self.ffn(ffn_inputs)
        if isinstance(result, tuple):
            ffn_output, auxiliary = result
            if not isinstance(auxiliary, dict):
                raise TypeError("an auxiliary FFN must return (Tensor, dict)")
        else:
            ffn_output = result
            auxiliary = {}
        hidden = hidden + self.ffn_dropout(ffn_output)

        if not collect_router_stats:
            auxiliary = {key: value for key, value in auxiliary.items() if key != "routing_stats"}
        return hidden, auxiliary


def _dense_ffn_factory(_: int, config: ModelConfig) -> nn.Module:
    return DenseFFN(config.d_model, config.d_ff, config.dropout)


class LanguageModel(nn.Module):
    """Four-block, pre-normalized causal decoder with tied token weights.

    ``ffn_factory`` is the intended Person 2 extension point. It receives a
    zero-based block index and may return a dense or MoE FFN. An MoE FFN should
    return ``(output, auxiliary)`` where differentiable scalar auxiliary losses
    use ``balance_loss`` and ``router_z_loss`` keys.
    """

    def __init__(
        self,
        config: ModelConfig,
        vocab_size: int,
        max_sequence_length: int,
        ffn_factory: FFNFactory | None = None,
    ) -> None:
        super().__init__()
        if config.model_type != "dense" and ffn_factory is None:
            raise ValueError("non-dense models require an ffn_factory supplied by the MoE implementation")
        self.config = config
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(max_sequence_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        factory = ffn_factory or _dense_ffn_factory
        self.blocks = nn.ModuleList(
            TransformerBlock(config, factory(layer_index, config))
            for layer_index in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: Tensor, collect_router_stats: bool = False) -> tuple[Tensor, Auxiliary]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        batch, sequence = tokens.shape
        if sequence > self.max_sequence_length:
            raise ValueError(f"sequence length {sequence} exceeds {self.max_sequence_length}")

        positions = torch.arange(sequence, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        hidden = self.embedding_dropout(hidden)

        balance_losses: list[Tensor] = []
        z_losses: list[Tensor] = []
        routing_stats: dict[str, Any] = {}
        for layer_index, block in enumerate(self.blocks):
            hidden, block_aux = block(hidden, collect_router_stats=collect_router_stats)
            if "balance_loss" in block_aux:
                balance_losses.append(block_aux["balance_loss"])
            if "router_z_loss" in block_aux:
                z_losses.append(block_aux["router_z_loss"])
            if collect_router_stats and "routing_stats" in block_aux:
                routing_stats[f"layer_{layer_index + 1}"] = block_aux["routing_stats"]

        hidden = self.final_norm(hidden)
        logits = F.linear(hidden, self.token_embedding.weight)
        zero = logits.new_zeros(())
        auxiliary: Auxiliary = {
            "balance_loss": torch.stack(balance_losses).mean() if balance_losses else zero,
            "router_z_loss": torch.stack(z_losses).mean() if z_losses else zero,
            "routing_stats": routing_stats,
        }
        return logits, auxiliary

    def unique_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_model(config):
    """Single construction point shared by training, evaluation and checks.

    Person 2 supplies an ffn_factory here for config.model.model_type == "moe".
    """
    return LanguageModel(config.model, config.data.vocab_size, config.data.context_length)
