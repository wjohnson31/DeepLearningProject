import copy
import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from moe_transformer.config import load_config
from moe_transformer.data import DeterministicBatcher, TokenWindows, validate_processed_data
from moe_transformer.evaluate import evaluate
from moe_transformer.model import DenseFFN, LanguageModel
from moe_transformer.prepare_data import (new_counts, parse_stories, prepare, train_tokenizer,
                                          unique_stories)
from moe_transformer.runtime import load_checkpoint, seed_all
from moe_transformer.schedules import balancing_coefficient
from moe_transformer.train import make_optimizer, train, training_update


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(2)
    config = load_config("configs/smoke.yaml")
    config.data.output_dir = str(tmp_path / "data")
    config.data.target_train_tokens = 512
    config.data.target_dev_tokens = 113  # Deliberately uneven final evaluation batch.
    config.data.target_test_tokens = 128
    config.model.dropout = .1
    config.training.max_steps = 6
    config.training.eval_interval = 3
    prepare(config, "synthetic")
    return config


def test_story_parser_dedup_and_overlap():
    rows = list(parse_stories(["Hello  world", "<|endoftext|>", "Hello world",
                              "<|endoftext|>", "Another story", "<|endoftext|>"]))
    counts = new_counts()
    stories = list(unique_stories(rows, counts))
    assert len(stories) == 2 and counts["duplicates_removed"] == 1
    counts = new_counts()
    assert len(list(unique_stories(rows, counts, {stories[0]["id"]}))) == 1
    assert counts["train_overlap_removed"] == 2


def test_byte_level_alphabet_covers_unseen_characters():
    stories = [{"text": "A little cat sees a dog. " * 20}]
    tokenizer = train_tokenizer(stories, 270)
    ids = tokenizer.encode("Unseen characters: ☃ café").ids
    assert tokenizer.token_to_id("<unk>") not in ids
    assert tokenizer.get_vocab_size() == 270


def test_batches_and_integrity(config):
    windows = TokenWindows(f"{config.data.output_dir}/train.npy", 16)
    with pytest.raises(ValueError):
        DeterministicBatcher(windows, len(windows) + 1, 0)
    first, second = DeterministicBatcher(windows, 4, 12), DeterministicBatcher(windows, 4, 12)
    for _ in range(20):  # Cross epoch boundaries.
        x, y = first.next()
        assert torch.equal(x, second.next()[0])
        assert torch.equal(x[:, 1:], y[:, :-1])
    state = first.state_dict()
    expected = first.next()[0]
    second.load_state_dict(state)
    assert torch.equal(expected, second.next()[0])
    path = f"{config.data.output_dir}/train.npy"
    values = np.load(path)
    values[0] = 0 if values[0] != 0 else 1
    np.save(path, values)
    with pytest.raises(ValueError, match="checksum"):
        validate_processed_data(config.data.output_dir)


def test_causality_aux_and_tied_parameters(config):
    model = LanguageModel(config.model, 64, 16).eval()
    x = torch.randint(0, 64, (2, 16))
    y = x.clone()
    y[:, 8:] = (y[:, 8:] + 1) % 64
    logits, aux = model(x)
    torch.testing.assert_close(logits[:, :8], model(y)[0][:, :8], atol=1e-6, rtol=0)
    assert aux["balance_loss"] == 0 and aux["routing_stats"] == {}
    # No separate output projection parameter: head uses embedding.weight.
    assert sum("token_embedding.weight" == name for name, _ in model.named_parameters()) == 1
    opt = make_optimizer(model, config)
    no_decay = {id(p) for p in opt.param_groups[1]["params"]}
    assert id(model.final_norm.weight) in no_decay
    assert id(model.blocks[0].ffn.up.bias) in no_decay


def test_aux_losses_average_only_contributing_layers(config):
    class StubExpert(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = nn.Parameter(torch.tensor(float(value)))

        def forward(self, x):
            return x * self.value, {"balance_loss": self.value, "router_z_loss": self.value**2,
                                     "routing_stats": torch.ones(4).detach()}

    def factory(index, settings):
        return StubExpert(index) if index in (1, 3) else DenseFFN(settings.d_model, settings.d_ff, 0)

    model = LanguageModel(config.model, 64, 16, factory)
    _, aux = model(torch.ones(2, 16, dtype=torch.long), collect_router_stats=True)
    assert float(aux["balance_loss"].detach()) == 2.
    assert float(aux["router_z_loss"].detach()) == 5.
    aux["balance_loss"].backward()
    assert model.blocks[1].ffn.value.grad == .5
    assert set(aux["routing_stats"]) == {"layer_2", "layer_4"}


def test_token_weighted_evaluation(config):
    model = LanguageModel(config.model, 64, 16)
    windows = TokenWindows(f"{config.data.output_dir}/dev.npy", 16)
    result = evaluate(model, windows, 4, torch.device("cpu"))
    model.eval()
    nll = 0.
    with torch.no_grad():
        for index in range(len(windows)):
            x, y = windows.get([index])
            nll += float(F.cross_entropy(model(x)[0].reshape(-1, 64), y.reshape(-1), reduction="sum"))
    assert result["target_tokens"] == len(windows) * 16
    assert result["ce"] == pytest.approx(nll / result["target_tokens"], abs=1e-6)
    assert result["perplexity"] == pytest.approx(math.exp(result["ce"]))


def test_resume_matches_uninterrupted_with_dropout(config, tmp_path):
    device = torch.device("cpu")
    full = train(config, tmp_path / "full", device)
    train(config, tmp_path / "prefix", device, stop_after=3)
    resumed = train(config, tmp_path / "resumed", device, resume=tmp_path / "prefix/last.pt")
    full_state = load_checkpoint(tmp_path / "full/last.pt")
    resumed_state = load_checkpoint(tmp_path / "resumed/last.pt")
    for key in full_state["model"]:
        torch.testing.assert_close(full_state["model"][key], resumed_state["model"][key], rtol=0, atol=0)
    assert full["dev_ce"] == resumed["dev_ce"]
    assert full_state["batcher"] == resumed_state["batcher"]
    assert full_state["evaluations"] == resumed_state["evaluations"]
    changed = copy.deepcopy(config)
    changed.training.max_steps = 10
    with pytest.raises(ValueError, match="configuration differs"):
        train(changed, tmp_path / "bad_resume", device, resume=tmp_path / "prefix/last.pt")


def test_balancing_schedule():
    assert balancing_coefficient(0, 5000, "scheduled") == .01
    assert balancing_coefficient(2500, 5000, "scheduled") == pytest.approx(.001)
    assert balancing_coefficient(5000, 5000, "scheduled") == pytest.approx(.001)


def test_accumulated_gradient_matches_full_batch(config):
    config.model.dropout = 0
    seed_all(0)
    model = LanguageModel(config.model, 64, 16)
    reference = copy.deepcopy(model)
    x, y = torch.randint(0, 64, (8, 16)), torch.randint(0, 64, (8, 16))
    for start in (0, 4):
        logits, _ = model(x[start:start + 4])
        (F.cross_entropy(logits.reshape(-1, 64), y[start:start + 4].reshape(-1)) / 2).backward()
    F.cross_entropy(reference(x)[0].reshape(-1, 64), y.reshape(-1)).backward()
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-7, rtol=1e-4)


def test_overflow_retry_replays_same_batch_and_dropout(config):
    # Emulate an overflowing first attempt to test retry bookkeeping on CPU.
    # Actual CUDA autocast/scaling still needs the Colab checks.
    class EmulatedScaler:
        def __init__(self):
            self.attempts = 0
            self.value = 65536.

        def is_enabled(self):
            return True

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            if self.attempts == 0:
                optimizer.param_groups[0]["params"][0].grad.fill_(float("inf"))

        def get_scale(self):
            return self.value

        def update(self, new_scale=None):
            self.attempts += 1
            if new_scale is not None:
                self.value = new_scale

        def step(self, optimizer):
            optimizer.step()

    def setup():
        seed_all(42)
        model = LanguageModel(config.model, 64, 16)
        batcher = DeterministicBatcher(TokenWindows(f"{config.data.output_dir}/train.npy", 16), 4, 0)
        return model, make_optimizer(model, config), batcher

    device = torch.device("cpu")
    expected, optimizer, batcher = setup()
    training_update(expected, optimizer, torch.amp.GradScaler("cuda", enabled=False),
                    batcher, config, device, 0.)
    expected_position = batcher.state_dict()
    actual, optimizer, batcher = setup()
    scaler = EmulatedScaler()
    _, _, retries = training_update(actual, optimizer, scaler, batcher, config, device, 0.)
    assert retries == 1 and scaler.value == 32768.
    assert batcher.state_dict() == expected_position
    for p, q in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_last_checkpoint_recovers_missing_best(config, tmp_path):
    device = torch.device("cpu")
    train(config, tmp_path / "prefix", device, stop_after=3)
    (tmp_path / "prefix/best.pt").unlink()
    result = train(config, tmp_path / "resumed", device, resume=tmp_path / "prefix/last.pt")
    best = load_checkpoint(tmp_path / "resumed/best.pt")
    assert best["checkpoint_kind"] == "evaluation"
    assert best["step"] == result["selected_checkpoint_step"]
    with pytest.raises(ValueError, match="evaluation-only"):
        train(config, tmp_path / "invalid", device, resume=tmp_path / "resumed/best.pt")
