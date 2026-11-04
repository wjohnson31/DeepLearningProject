"""Original TinyStories release: whole-story selection before packing.

Public source is streamed only as far as needed. The revision, source indices,
normalized-text hashes and removal counts are preserved with the token caches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import requests

from .config import ExperimentConfig, load_config
from .data import save_token_array, sha256_file, validate_processed_data

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
SOURCE_FILES = {"train": "TinyStories-train.txt", "validation": "TinyStories-valid.txt"}


def _normalized_story(text: str) -> str:
    return " ".join(text.strip().split())


def _story_id(text: str) -> str:
    return hashlib.sha256(_normalized_story(text).encode("utf-8")).hexdigest()


def parse_stories(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    buffer: list[str] = []
    index = 0
    for line in lines:
        if line.strip() == "<|endoftext|>":
            yield {"text": "\n".join(buffer), "source_index": index}
            index += 1
            buffer.clear()
        else:
            buffer.append(line)
    if buffer and "".join(buffer).strip():
        yield {"text": "\n".join(buffer), "source_index": index}


def source_rows(config: ExperimentConfig, split: str) -> Iterator[dict[str, Any]]:
    url = (f"https://huggingface.co/datasets/{config.data.dataset_id}/resolve/"
           f"{config.data.revision}/{SOURCE_FILES[split]}")
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        response.encoding = "utf-8"
        yield from parse_stories(response.iter_lines(decode_unicode=True))


def unique_stories(rows, counts, excluded=None):
    excluded = set(excluded or ())
    seen: set[str] = set()
    for row in rows:
        counts["scanned"] += 1
        text = _normalized_story(row["text"])
        if not text:
            counts["empty_removed"] += 1
            continue
        identifier = _story_id(text)
        if identifier in excluded:
            counts["train_overlap_removed"] += 1
            continue
        if identifier in seen:
            counts["duplicates_removed"] += 1
            continue
        seen.add(identifier)
        yield {"id": identifier, "text": text, "source_index": row["source_index"]}


def new_counts():
    return dict(scanned=0, empty_removed=0, duplicates_removed=0, train_overlap_removed=0)


def train_tokenizer(stories, vocab_size):
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                         special_tokens=SPECIAL_TOKENS,
                         initial_alphabet=ByteLevel.alphabet(), show_progress=False)
    tokenizer.train_from_iterator((story["text"] for story in stories), trainer=trainer,
                                  length=len(stories))
    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError("Not enough training text to fit the requested vocabulary")
    return tokenizer


def select_to_target(tokenizer, stories, target):
    selected, tokens = [], []
    eos = tokenizer.token_to_id("<eos>")
    for story in stories:
        selected.append(story)
        tokens.extend(tokenizer.encode(story["text"]).ids)
        tokens.append(eos)
        if len(tokens) >= target:
            return selected, tokens
    raise ValueError(f"Source ended at {len(tokens)} tokens before target {target}")


def _prepare_hf(config, destination):
    counts = {"train": new_counts(), "validation": new_counts()}
    stream = source_rows(config, "train")
    candidates = []
    characters = 0
    # A deterministic pool of training-only text used for vocabulary fitting.
    # The retained training prefix is subsequently chosen with this frozen BPE.
    # Metadata explicitly records BOTH pools.
    try:
        for story in unique_stories(stream, counts["train"]):
            candidates.append(story)
            characters += len(story["text"])
            if characters >= config.data.target_train_tokens * 5:
                break
    finally:
        stream.close()
    print(f"Fitting BPE on {len(candidates):,} unique training-source stories", flush=True)
    tokenizer = train_tokenizer(candidates, config.data.vocab_size)
    train, train_tokens = select_to_target(tokenizer, iter(candidates), config.data.target_train_tokens)
    # Refit on exactly the retained training stories. Actual final count is
    # approximately the target and is reported without trimming a story.
    tokenizer = train_tokenizer(train, config.data.vocab_size)
    train_tokens = []
    for story in train:
        train_tokens.extend(tokenizer.encode(story["text"]).ids)
        train_tokens.append(tokenizer.token_to_id("<eos>"))
    tokenizer.save(str(destination / "tokenizer.json"))
    train_ids = {story["id"] for story in train}
    stream = source_rows(config, "validation")
    validation = unique_stories(stream, counts["validation"], train_ids)
    try:
        dev, dev_tokens = select_to_target(tokenizer, validation, config.data.target_dev_tokens)
        test, test_tokens = select_to_target(tokenizer, validation, config.data.target_test_tokens)
    finally:
        stream.close()
    stories = {"train": train, "dev": dev, "test": test}
    splits = {}
    for split, tokens in (("train", train_tokens), ("dev", dev_tokens), ("test", test_tokens)):
        splits[split] = save_token_array(destination / f"{split}.npy", tokens)
    return {
        "source": config.data.dataset_id, "revision": config.data.revision,
        "source_files": SOURCE_FILES, "license": "cdla-sharing-1.0",
        "selection": "deterministic source-order prefixes of whole normalized stories",
        "normalization": "strip and collapse whitespace, preserve case",
        "deduplication_counts": counts,
        "selection_pool_stories": len(candidates),
        "final_tokenizer_training_stories": len(train),
        "tokenizer": {"path": "tokenizer.json", "sha256": sha256_file(destination / "tokenizer.json")},
        "splits": splits,
        "story_counts": {key: len(value) for key, value in stories.items()},
        "story_ids": {key: [s["id"] for s in value] for key, value in stories.items()},
        "source_indices": {key: [s["source_index"] for s in value] for key, value in stories.items()},
    }


def _prepare_synthetic(config, destination):
    generator = np.random.default_rng(config.seed)
    splits = {}
    for index, (split, size) in enumerate(zip(
        ("train", "dev", "test"),
        (config.data.target_train_tokens, config.data.target_dev_tokens, config.data.target_test_tokens),
    )):
        tokens = (np.arange(size, dtype=np.int32) + index * 7) % config.data.vocab_size
        noise = generator.random(size) < .03
        tokens[noise] = generator.integers(0, config.data.vocab_size, int(noise.sum()), dtype=np.int32)
        splits[split] = save_token_array(destination / f"{split}.npy", tokens)
    return {
        "source": "synthetic-smoke-only", "tokenizer": None,
        "splits": splits, "story_counts": {key: 1 for key in splits},
        "story_ids": {key: [_story_id(f"synthetic-{key}")] for key in splits},
    }


def prepare(config: ExperimentConfig, source: str) -> Path:
    destination = Path(config.data.output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; reuse it or select a new output path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a validated directory. No destructive --force operation.
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        metadata = _prepare_synthetic(config, staging) if source == "synthetic" else _prepare_hf(config, staging)
        metadata.update({
            "vocab_size": config.data.vocab_size, "context_length": config.data.context_length,
            "seed": config.seed, "schema_version": 1,
            "packing": "EOS between stories; windows stride context_length; incomplete final window dropped",
            "targets": {"train": config.data.target_train_tokens, "dev": config.data.target_dev_tokens,
                        "test": config.data.target_test_tokens},
        })
        for record in metadata["splits"].values():
            windows = (record["tokens"] - 1) // config.data.context_length
            record["windows"] = windows
            record["evaluated_target_tokens"] = windows * config.data.context_length
            record["unused_tail_tokens"] = record["tokens"] - (windows * config.data.context_length + 1)
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        validate_processed_data(staging)
        staging.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/milestone.yaml")
    parser.add_argument("--source", choices=("hf", "synthetic"), default="hf")
    args = parser.parse_args()
    config = load_config(args.config)
    if (args.source == "synthetic") != (config.data.dataset_id == "synthetic"):
        parser.error("synthetic source requires a synthetic config, and vice versa")
    print(f"Validated processed data at {prepare(config, args.source)}")


if __name__ == "__main__":
    main()
