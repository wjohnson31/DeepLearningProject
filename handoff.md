# Person 1 handoff: dense TinyStories baseline

Run status: **complete**. Source: roneneldan/TinyStories.
The MoE implementation and MoE experiments belong to Person 2.

## Measured baseline

| Field | Recorded value |
|---|---|
| Unique trainable parameters | 2,590,848 |
| Completed / planned updates | 5000 / 5000 |
| Target-token exposures | 20,480,000 |
| Selected checkpoint step | 5000 |
| Development CE (nats) | 2.43751250 |
| Development perplexity | 11.44453700 |
| Analytical forward matmul FLOPs/token | 5,505,024 |
| Device / precision | cpu / fp32 |
| Full-update throughput, latest session | 10028.27 tokens/s |
| Training time across resumed segments | 2006.85 seconds |
| Recorded elapsed time across resumed segments | 2058.41 seconds |
| Peak CUDA training memory | None |

Throughput excludes the first 20 updates of the latest session and
excludes development evaluation/checkpoint writes. Elapsed time is recorded
immediately before the final checkpoint writes; it excludes process setup
and the initial step-0 development evaluation. Null CUDA memory means
unavailable on the recorded device, not zero memory use.

## Data and protocol

| Split | Retained stories | Tokens | Full 128-target windows |
|---|---:|---:|---:|
| train | 45,852 | 10,000,138 | 78,126 |
| dev | 1,047 | 200,143 | 1,563 |
| test | 1,042 | 200,132 | 1,563 |

Dataset revision: f54c09fd23315a6f9c86f9dc80f725de7d8f9c64.
Metadata identity: 6f7ac8da8f4244be8bd887febb9b5c652aba14501b2a24c9bdfa759f75405809.
Code identity for this run: 43024fa4b0e8e4e0cf4c6d2c7565b1d79fa97efc7402f61a5d310775a20e93b1.
The final test set was created and integrity-checked, but never evaluated.
The tokenizer has 4096 entries including four special tokens; final BPE
training uses only the retained training stories. Every story has a saved
normalized-text SHA-256 and original source index in metadata.json.

Architecture: four pre-LayerNorm blocks, width 192, four heads, FFN width
768, context 128, learned positions, tied input/output weights, dropout 0.1.
AdamW: peak LR 0.001, final target 0.0001, cosine schedule, weight decay 0.1,
betas (0.9, 0.95), epsilon 1e-8, clipping 1.0. Microbatch 16 with two
accumulation steps gives 4096 predictions per update. Seed 0; shuffled
window order uses seed + epoch. The exact warmup/horizon and precision
are recorded in runs/D1_cpu_full/config.yaml.

Development evaluation steps in this run: [0, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000].
Step 0 is diagnostic only. Match the listed trained evaluation points
in the paired MoE run, including any extra pilot-stop evaluation.

## Exact commands and artifacts

From the project root with its environment installed:

~~~bash
export PYTHONPATH="$PWD/src"
python -m pytest -q
python -m moe_transformer.checks --config configs/milestone.yaml --device cpu --output artifacts/person1/real_data_checks.json
python -m moe_transformer.evaluate --checkpoint runs/D1_cpu_full/best.pt --data-dir data/processed/tinystories_10m_bpe4096 --device cpu --output runs/D1_cpu_full/reproduced_dev.json
python -m moe_transformer.plotting runs/D1_cpu_full --output artifacts/person1/figures
~~~

This local baseline was produced with the following pilot and resume commands:

~~~bash
python -m moe_transformer.train --config configs/milestone.yaml --run-dir runs/D1_cpu_pilot --device cpu --precision fp32 --stop-after 200
python -m moe_transformer.train --config runs/D1_cpu_pilot/config.yaml --run-dir runs/D1_cpu_full --device cpu --resume runs/D1_cpu_pilot/last.pt
~~~

Preparation in a fresh directory (the bundle already contains its outputs):

~~~bash
python -m moe_transformer.prepare_data --config configs/milestone.yaml
~~~

Run artifacts: runs/D1_cpu_full/best.pt, last.pt, results.json, evaluations.json,
train.jsonl, config.yaml, environment.json, code_identity.json, and source/.
Frozen tokenizer/caches: data/processed/tinystories_10m_bpe4096/.
The source/ snapshot and code hashes preserve the code used even without a
Git commit. Checkpoints include optimizer, scaler, RNG and sampler states.
The LR schedule is reconstructed from completed step and frozen config.

Resume a partial run into a fresh directory:

~~~bash
python -m moe_transformer.train --config PATH_TO_PARTIAL_RUN/config.yaml --run-dir runs/D1_resumed --resume PATH_TO_PARTIAL_RUN/last.pt --device cpu
~~~

For a one-update handoff acceptance check, use a checkpoint whose completed
step is below the planned horizon and pass --stop-after CHECKPOINT_STEP_PLUS_ONE.
Keep the matching best.pt beside last.pt. A completed run deliberately refuses
to extend its frozen horizon; use the saved pilot for this check.

The supplied CPU pilot supports this concrete one-update command:

~~~bash
python -m moe_transformer.train --config runs/D1_cpu_pilot/config.yaml --device cpu --run-dir runs/karan_acceptance --resume runs/D1_cpu_pilot/last.pt --stop-after 201
~~~

## Person 2 extension points

- Add the MoE FFN and its factory in model.py; build_model(config) is the
  common entry point used by training, evaluation and checks.
- Replace only zero-based block indices 1 and 3. The FFN receives [B,S,d].
  Return (output, auxiliary) with differentiable scalar balance_loss and
  router_z_loss; diagnostic routing_stats must be detached assignment counts
  or a dict with assignment_counts. DenseFFN returns just its output tensor.
- LanguageModel returns logits [B,S,V] and auxiliary fields. Routing losses
  are averaged only over layers that supply them. The shared trainer already
  adds alpha * balance_loss + z_coefficient * router_z_loss.
- schedules.py contains fixed/zero/scheduled alpha policies. No actual
  MoE training or schedule-in-MoE smoke result is claimed here.
- Add router and active-expert terms to runtime.forward_flops before using
  the trainer to report MoE compute; the dense-only estimate deliberately
  raises for MoE until that formula is implemented.
- Preserve data, tokenizer, architecture, precision, effective batch, seed,
  exposure budget, and checkpoint-selection protocol for the paired study.

## Environment, deviations and remaining work

Python 3.12.11; platform macOS-15.0-arm64-arm-64bit.
Versions: {"PyYAML": "6.0.3", "matplotlib": "3.11.2", "numpy": "2.5.4", "requests": "2.34.2", "tokenizers": "0.23.3", "torch": "2.15.0"}.

This measured run used CPU/FP32 rather than the proposal's T4/FP16.
Do not present its timing as GPU throughput. Local GPU time spent: 0.
For a CUDA comparison, run the included Colab notebook for both D1 and
M1 under the same device/precision and frozen budget. Alternatively,
use this recorded CPU/FP32 protocol for both models.
The arithmetic estimate excludes scalar normalization/activation/softmax
operations; document this convention and extend it consistently for routing.
Source-prefix selection, whitespace normalization, dropped final windows,
and the exact dropout/weight-decay conventions are in docs/person1_guide.md.
This is one seed. Its development result is baseline evidence and does not
establish any MoE or balancing-schedule benefit.

The full-budget dense run is finished. Verify reproduced_dev.json and the correctness records, then use these measured artifacts in the report.
Karan still owns MoE integration and diagnostics. Both authors must review
the final technical claims and write the report together.
