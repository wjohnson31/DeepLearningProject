"""Audit the measured dense baseline and export its report-ready evidence."""
import argparse
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from moe_transformer.config import _construct
from moe_transformer.data import TokenWindows, sha256_file, validate_processed_data
from moe_transformer.evaluate import evaluate
from moe_transformer.model import build_model
from moe_transformer.runtime import load_checkpoint, select_device, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/D1_cpu_full")
    parser.add_argument("--pilot-dir", default="runs/D1_cpu_pilot")
    parser.add_argument("--acceptance-dir", default="runs/handoff_acceptance")
    parser.add_argument("--output-dir", default="artifacts/person1")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run, pilot, output = map(Path, (args.run_dir, args.pilot_dir, args.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    test_report = output / "test_results.xml"
    subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={test_report}"], check=True)
    suites = list(ET.parse(test_report).getroot().iter("testsuite"))
    tests_passed = sum(int(s.attrib["tests"]) - int(s.attrib.get("skipped", 0)) for s in suites)
    assert tests_passed > 0
    result = json.loads((run / "results.json").read_text())
    assert result["status"] == "complete", "Full-budget baseline is not finished"
    saved = load_checkpoint(run / "best.pt")
    last = load_checkpoint(run / "last.pt")
    config = _construct(saved["config"])
    metadata = validate_processed_data(config.data.output_dir)
    assert saved["data_identity"] == sha256_file(Path(config.data.output_dir) / "metadata.json")
    assert saved["step"] == result["selected_checkpoint_step"]
    assert last["step"] == result["completed_steps"] == config.training.max_steps
    history = json.loads((run / "evaluations.json").read_text())
    candidates = [row for row in history if row["step"] > 0]
    best = min(candidates, key=lambda row: row["ce"])
    assert best["step"] == saved["step"] and best["ce"] == result["dev_ce"]
    rows = []
    for directory in (pilot, run):
        rows.extend(json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines())
    assert [row["step"] for row in rows] == list(range(1, last["step"] + 1))
    exposures = (config.training.micro_batch_size * config.training.gradient_accumulation_steps
                 * config.data.context_length)
    for row in rows:
        assert row["token_exposures"] == row["step"] * exposures
        assert all(math.isfinite(row[key]) for key in ("ce", "total_loss", "gradient_norm"))
    assert rows[-1]["token_exposures"] == result["token_exposures"]
    device = select_device(result["environment"]["device"])
    model = build_model(config).to(device)
    model.load_state_dict(saved["model"])
    assert model.unique_parameter_count() == result["parameters"]
    measured = evaluate(model, TokenWindows(Path(config.data.output_dir) / "dev.npy",
                        config.data.context_length), config.training.micro_batch_size,
                        device, config.training.precision)
    difference = abs(measured["ce"] - result["dev_ce"])
    assert difference < 1e-6
    write_json(run / "reproduced_dev.json", dict(measured, step=saved["step"], split="dev",
               precision=config.training.precision, device=str(device)))
    # Check actual one-update acceptance against the same update in the full run.
    acceptance_path = Path(args.acceptance_dir) / "train.jsonl"
    if acceptance_path.exists():
        acceptance = json.loads(acceptance_path.read_text().splitlines()[0])
        write_json(output / "acceptance_update.json", acceptance)
    else:
        acceptance = json.loads((output / "acceptance_update.json").read_text())
    original = next(row for row in rows if row["step"] == acceptance["step"])
    assert acceptance["ce"] == original["ce"]
    write_json(output / "verification.json", {
        "status": "passed", "pipeline_tests_passed": tests_passed,
        "test_report": str(test_report),
        "checkpoint_dev_ce_absolute_difference": difference,
        "one_update_resume_ce_identical": True, "acceptance_update": acceptance["step"],
        "training_log_updates": len(rows), "token_exposures": rows[-1]["token_exposures"],
        "best_checkpoint_sha256": sha256_file(run / "best.pt"),
        "last_checkpoint_sha256": sha256_file(run / "last.pt"),
        "data_identity": saved["data_identity"], "model_parameters": model.unique_parameter_count(),
        "final_test_evaluated": False, "cuda_execution_tested": device.type == "cuda",
    })
    write_json(output / "baseline_results.json", result)
    write_json(output / "baseline_evaluations.json", history)
    write_json(output / "dataset_summary.json", {
        key: value for key, value in metadata.items() if key not in ("story_ids", "source_indices")
    })
    prose = (
        "# Measured dense baseline evidence\n\n"
        f"D1 has {result['parameters']:,} unique trainable parameters. It was trained from "
        f"scratch on {metadata['story_counts']['train']:,} TinyStories stories "
        f"({metadata['splits']['train']['tokens']:,} byte-level BPE tokens), using a frozen "
        f"{metadata['vocab_size']}-entry vocabulary. Training completed "
        f"{result['completed_steps']:,} updates with {result['token_exposures']:,} "
        f"target-token exposures. The lowest development CE was {result['dev_ce']:.6f} nats "
        f"(perplexity {result['dev_perplexity']:.6f}) at update "
        f"{result['selected_checkpoint_step']:,}. Development evaluation covered "
        f"{measured['target_tokens']:,} target tokens. Reloading the selected checkpoint "
        f"reproduced CE with absolute difference {difference:.3g}.\n\n"
        f"The recorded environment was {result['environment']['device']}, "
        f"{result['environment']['precision']}, with four PyTorch CPU threads. "
        f"Measured training time across the pilot and resumed run was "
        f"{result['training_seconds'] / 60:.2f} minutes. Throughput in the resumed "
        f"session, excluding its first 20 updates, was "
        f"{result['session_profile_tokens_per_second']:,.1f} target tokens/second. "
        f"The analytical forward matrix-operation estimate is "
        f"{result['forward_matmul_flops_per_token']:,} FLOPs/token, counting two operations "
        "per multiply-add and excluding scalar nonlinearities and normalization. "
        "GPU throughput and peak GPU memory were not measured.\n\n"
        "These are single-seed dense-baseline results. They do not establish an MoE "
        "improvement, a balancing-schedule benefit, or statistical robustness. "
        "The final-test partition has not been evaluated. A paired MoE comparison "
        "must preserve this protocol or rerun the dense control with the revised one.\n\n"
        "Raw evidence: baseline_results.json, baseline_evaluations.json, "
        "verification.json, dataset_summary.json, and figures/learning_curve.png. "
        "All reported metrics originate from the saved checkpoints and logs.\n"
    )
    (output / "baseline_summary.md").write_text(prose)
    print(f"PASS: {len(rows)} updates, checkpoint CE reproduced, one-update acceptance matches")


if __name__ == "__main__":
    main()
