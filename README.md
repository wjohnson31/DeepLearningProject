# TinyStories dense baseline — Person 1

This repository implements the first half of the supplied MoE milestone roadmap:
data preparation, a dense decoder, training/evaluation, resumable checkpoints,
correctness checks, and a handoff for the MoE implementation.

**Start with handoff.md** for the measured baseline, current completion status,
and the commands Karan needs. Read docs/person1_guide.md to understand your work.
Use notebooks/Person1_Colab.ipynb if you choose to run the GPU protocol.
The portable bundles include the fixed real-data subset, tokenizer, code,
tests and instructions.

## Local setup

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
export PYTHONPATH="$PWD/src"
pytest
~~~

The explicit PYTHONPATH also supports environments that restrict editable
installation paths. Colab sets it automatically.

## Real data and correctness checks

The prepared data is already included in the bundle. To rebuild in a fresh copy:

~~~bash
python -m moe_transformer.prepare_data --config configs/milestone.yaml
python -m moe_transformer.checks --config configs/milestone.yaml --device cpu \
  --output artifacts/person1/real_data_checks.json
~~~

Preparation refuses to overwrite an existing directory. Reuse the frozen
artifacts for comparisons. metadata.json records the original release,
revision, counts, hashes, tokenizer identity and split assignments.

## CUDA training, resuming and evaluation

~~~bash
# Pilot uses the full planned horizon but stops after update 200.
python -m moe_transformer.train --config configs/milestone.yaml \
  --device cuda --run-dir runs/D1_cuda_pilot --stop-after 200

# Resume that exact protocol into a new output directory.
python -m moe_transformer.train --config configs/milestone.yaml \
  --device cuda --run-dir runs/D1_cuda_full --resume runs/D1_cuda_pilot/last.pt

python -m moe_transformer.evaluate --checkpoint runs/D1_cuda_full/best.pt \
  --device cuda --output runs/D1_cuda_full/reproduced_dev.json
python -m moe_transformer.plotting runs/D1_cuda_full --output artifacts/figures
~~~

For the preplanned fallback, start a fresh run with
--config configs/milestone_2500.yaml. Do not resume the 5000-step pilot under
that shorter schedule.

CPU diagnostic runs add --device cpu --precision fp32. Their throughput and
memory measurements are not CUDA measurements.

## Artifacts to hand over

- Source code, exact chosen config, environment.json, code_identity.json.
- The entire processed-data directory, including tokenizer and metadata.
- Run directory containing best.pt, last.pt, train.jsonl, evaluations.json,
  results.json and a snapshot of the source used.
- Reproduced development evaluation, learning curve, and correctness records.
- handoff.md, updated with the completed GPU run's metrics and paths.

See docs/person1_guide.md for the data/model/training explanation and
handoff.md for the current measured status.

## Quick synthetic check (optional)

~~~bash
python -m moe_transformer.prepare_data --config configs/smoke.yaml --source synthetic
python -m moe_transformer.train --config configs/smoke.yaml --device cpu \
  --run-dir runs/synthetic_check
~~~

Synthetic and repeated-batch checks are implementation checks only.

## Build a portable bundle

~~~bash
python scripts/build_bundle.py
# To include a particular run in a handoff archive:
python scripts/build_bundle.py --output output/person1_handoff.zip \
  --include-run runs/D1_cuda_full
~~~
