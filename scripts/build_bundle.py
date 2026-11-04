"""Build a portable source/data bundle for Colab; optionally include run evidence."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/person1_colab_bundle.zip")
    parser.add_argument("--include-run", action="append", default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates = []
    for directory in ("src", "configs", "scripts", "tests", "notebooks", "docs",
                      "artifacts/person1", "data/processed/tinystories_10m_bpe4096"):
        candidates.extend((root / directory).rglob("*"))
    for name in ("README.md", "handoff.md", "pyproject.toml", ".gitignore"):
        candidates.append(root / name)
    for directory in args.include_run:
        path = (root / directory).resolve()
        if not path.is_relative_to(root / "runs"):
            raise ValueError("include-run must name a directory under this project's runs/")
        candidates.extend(path.rglob("*"))
    paths = sorted({p for p in candidates if p.is_file() and "__pycache__" not in p.parts})
    manifest = {}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            relative = path.relative_to(root).as_posix()
            content = path.read_bytes()
            manifest[relative] = hashlib.sha256(content).hexdigest()
            archive.writestr("DeepLearningProject/" + relative, content)
        archive.writestr("DeepLearningProject/bundle_manifest.json", json.dumps(manifest, indent=2))
    print(f"{output}: {len(paths)} files, {output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
