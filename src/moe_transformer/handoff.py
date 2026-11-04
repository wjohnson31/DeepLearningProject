"""Export measured Person 1 status and reproducible commands from a run."""
import argparse
import json
from pathlib import Path

from .data import load_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", default="data/processed/tinystories_10m_bpe4096")
    parser.add_argument("--output", default="handoff.md")
    args = parser.parse_args()
    run = Path(args.run_dir)
    result = json.loads((run / "results.json").read_text())
    evaluation_steps = [row["step"] for row in json.loads((run / "evaluations.json").read_text())]
    metadata = load_metadata(args.data_dir)
    env = result["environment"]
    is_complete = result["status"] == "complete"
    throughput = result["session_profile_tokens_per_second"]
    throughput_text = f"{throughput:.2f} tokens/s" if throughput is not None else "not measured (warmup only)"
    lines = [
        "# Person 1 handoff: dense TinyStories baseline", "",
        f"Run status: **{result['status']}**. Source: {result['source']}.",
        "The MoE implementation and MoE experiments belong to Person 2.", "",
        "## Measured baseline", "",
        "| Field | Recorded value |", "|---|---|",
        f"| Unique trainable parameters | {result['parameters']:,} |",
        f"| Completed / planned updates | {result['completed_steps']} / {result['planned_steps']} |",
        f"| Target-token exposures | {result['token_exposures']:,} |",
        f"| Selected checkpoint step | {result['selected_checkpoint_step']} |",
        f"| Development CE (nats) | {result['dev_ce']:.8f} |",
        f"| Development perplexity | {result['dev_perplexity']:.8f} |",
        f"| Analytical forward matmul FLOPs/token | {result['forward_matmul_flops_per_token']:,} |",
        f"| Device / precision | {env['device']} / {env['precision']} |",
        f"| Full-update throughput, latest session | {throughput_text} |",
        f"| Training time across resumed segments | {result['training_seconds']:.2f} seconds |",
        f"| Recorded elapsed time across resumed segments | {result['elapsed_seconds']:.2f} seconds |",
        f"| Peak CUDA training memory | {result['peak_training_cuda_memory_bytes']} |",
        "", "Throughput excludes the first 20 updates of the latest session and",
        "excludes development evaluation/checkpoint writes. Elapsed time is recorded",
        "immediately before the final checkpoint writes; it excludes process setup",
        "and the initial step-0 development evaluation. Null CUDA memory means",
        "unavailable on the recorded device, not zero memory use.", "",
        "## Data and protocol", "",
        "| Split | Retained stories | Tokens | Full 128-target windows |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train", "dev", "test"):
        stats = metadata["splits"][split]
        lines.append(f"| {split} | {metadata['story_counts'][split]:,} | {stats['tokens']:,} | {stats['windows']:,} |")
    lines += [
        "", f"Dataset revision: {metadata.get('revision', 'synthetic')}.",
        f"Metadata identity: {result['data_identity']}.",
        f"Code identity for this run: {result['code_identity']}.",
        "The final test set was created and integrity-checked, but never evaluated.",
        "The tokenizer has 4096 entries including four special tokens; final BPE",
        "training uses only the retained training stories. Every story has a saved",
        "normalized-text SHA-256 and original source index in metadata.json.", "",
        "Architecture: four pre-LayerNorm blocks, width 192, four heads, FFN width",
        "768, context 128, learned positions, tied input/output weights, dropout 0.1.",
        "AdamW: peak LR 0.001, final target 0.0001, cosine schedule, weight decay 0.1,",
        "betas (0.9, 0.95), epsilon 1e-8, clipping 1.0. Microbatch 16 with two",
        "accumulation steps gives 4096 predictions per update. Seed 0; shuffled",
        "window order uses seed + epoch. The exact warmup/horizon and precision",
        f"are recorded in {run}/config.yaml.", "",
        f"Development evaluation steps in this run: {evaluation_steps}.",
        "Step 0 is diagnostic only. Match the listed trained evaluation points",
        "in the paired MoE run, including any extra pilot-stop evaluation.", "",
        "## Exact commands and artifacts", "",
        "From the project root with its environment installed:", "",
        "~~~bash",
        "export PYTHONPATH=\"$PWD/src\"",
        "python -m pytest -q",
        "python -m moe_transformer.checks --config configs/milestone.yaml --device cpu "
        "--output artifacts/person1/real_data_checks.json",
        f"python -m moe_transformer.evaluate --checkpoint {run}/best.pt "
        f"--data-dir {args.data_dir} --device {env['device']} "
        f"--output {run}/reproduced_dev.json",
        f"python -m moe_transformer.plotting {run} --output artifacts/person1/figures",
        "~~~", "",
        *(["This local baseline was produced with the following pilot and resume commands:", "",
           "~~~bash",
           "python -m moe_transformer.train --config configs/milestone.yaml "
           "--run-dir runs/D1_cpu_pilot --device cpu --precision fp32 --stop-after 200",
           "python -m moe_transformer.train --config runs/D1_cpu_pilot/config.yaml "
           "--run-dir runs/D1_cpu_full --device cpu --resume runs/D1_cpu_pilot/last.pt",
           "~~~", ""]
          if env["device"].startswith("cpu") else
          ["This GPU run was produced with the Colab notebook. Its saved config",
           "and environment describe the actual protocol and assigned GPU.", ""]),
        "Preparation in a fresh directory (the bundle already contains its outputs):", "",
        "~~~bash",
        "python -m moe_transformer.prepare_data --config configs/milestone.yaml",
        "~~~", "",
        f"Run artifacts: {run}/best.pt, last.pt, results.json, evaluations.json,",
        "train.jsonl, config.yaml, environment.json, code_identity.json, and source/.",
        f"Frozen tokenizer/caches: {args.data_dir}/.",
        "The source/ snapshot and code hashes preserve the code used even without a",
        "Git commit. Checkpoints include optimizer, scaler, RNG and sampler states.",
        "The LR schedule is reconstructed from completed step and frozen config.", "",
        "Resume a partial run into a fresh directory:", "",
        "~~~bash",
        "python -m moe_transformer.train --config PATH_TO_PARTIAL_RUN/config.yaml "
        "--run-dir runs/D1_resumed --resume PATH_TO_PARTIAL_RUN/last.pt "
        f"--device {env['device']}",
        "~~~", "",
        "For a one-update handoff acceptance check, use a checkpoint whose completed",
        "step is below the planned horizon and pass --stop-after CHECKPOINT_STEP_PLUS_ONE.",
        "Keep the matching best.pt beside last.pt. A completed run deliberately refuses",
        "to extend its frozen horizon; use the saved pilot for this check.", "",
        *(["The supplied CPU pilot supports this concrete one-update command:", "",
           "~~~bash",
           "python -m moe_transformer.train --config runs/D1_cpu_pilot/config.yaml "
           "--device cpu --run-dir runs/karan_acceptance --resume runs/D1_cpu_pilot/last.pt "
           "--stop-after 201",
           "~~~", ""] if env["device"].startswith("cpu") else
          ["If no partial checkpoint is included, use the saved configuration with",
           "--stop-after 1 and a new run directory for a fresh one-update trainer check.", ""]),
        "## Person 2 extension points", "",
        "- Add the MoE FFN and its factory in model.py; build_model(config) is the",
        "  common entry point used by training, evaluation and checks.",
        "- Replace only zero-based block indices 1 and 3. The FFN receives [B,S,d].",
        "  Return (output, auxiliary) with differentiable scalar balance_loss and",
        "  router_z_loss; diagnostic routing_stats must be detached assignment counts",
        "  or a dict with assignment_counts. DenseFFN returns just its output tensor.",
        "- LanguageModel returns logits [B,S,V] and auxiliary fields. Routing losses",
        "  are averaged only over layers that supply them. The shared trainer already",
        "  adds alpha * balance_loss + z_coefficient * router_z_loss.",
        "- schedules.py contains fixed/zero/scheduled alpha policies. No actual",
        "  MoE training or schedule-in-MoE smoke result is claimed here.",
        "- Add router and active-expert terms to runtime.forward_flops before using",
        "  the trainer to report MoE compute; the dense-only estimate deliberately",
        "  raises for MoE until that formula is implemented.",
        "- Preserve data, tokenizer, architecture, precision, effective batch, seed,",
        "  exposure budget, and checkpoint-selection protocol for the paired study.", "",
        "## Environment, deviations and remaining work", "",
        f"Python {env['python']}; platform {env['platform']}.",
        f"Versions: {json.dumps(env['versions'], sort_keys=True)}.", "",
    ]
    if env["device"].startswith("cpu"):
        lines += [
            "This measured run used CPU/FP32 rather than the proposal's T4/FP16.",
            "Do not present its timing as GPU throughput. Local GPU time spent: 0.",
            "For a CUDA comparison, run the included Colab notebook for both D1 and",
            "M1 under the same device/precision and frozen budget. Alternatively,",
            "use this recorded CPU/FP32 protocol for both models.",
        ]
    lines += [
        "The arithmetic estimate excludes scalar normalization/activation/softmax",
        "operations; document this convention and extend it consistently for routing.",
        "Source-prefix selection, whitespace normalization, dropped final windows,",
        "and the exact dropout/weight-decay conventions are in docs/person1_guide.md.",
        "This is one seed. Its development result is baseline evidence and does not",
        "establish any MoE or balancing-schedule benefit.",
        "",
        ("The full-budget dense run is finished. Verify reproduced_dev.json and the"
         " correctness records, then use these measured artifacts in the report."
         if is_complete else
         "This run is preliminary. Complete the planned baseline or explicitly label"
         " its exposure budget and partial status in the report."),
        "Karan still owns MoE integration and diagnostics. Both authors must review",
        "the final technical claims and write the report together.", "",
    ]
    Path(args.output).write_text("\n".join(lines))
    print(f"Saved measured handoff to {args.output}")


if __name__ == "__main__":
    main()
