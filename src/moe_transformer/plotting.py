"""Plot only measured development evaluations from one or more runs."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_runs(run_dirs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    rows = []
    for directory in map(Path, run_dirs):
        result = json.loads((directory / "results.json").read_text())
        history = json.loads((directory / "evaluations.json").read_text())
        label = f"{directory.name} ({result['status']})"
        if result["source"] == "synthetic-smoke-only":
            label += " — synthetic check"
        ax.plot([row["token_exposures"] / 1e6 for row in history],
                [row["ce"] for row in history], marker="o", markersize=4, label=label)
        rows.append({k: v for k, v in result.items() if not isinstance(v, dict)})
    ax.set(xlabel="Training target-token exposures (millions)", ylabel="Development cross-entropy (nats)",
           title="Dense baseline learning trajectory")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(output / "learning_curve.png", dpi=180)
    plt.close(fig)
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", default="artifacts/figures")
    args = parser.parse_args()
    print(plot_runs(args.runs, args.output))


if __name__ == "__main__":
    main()
