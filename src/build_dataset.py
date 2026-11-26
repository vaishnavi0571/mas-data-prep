"""Build the leaves from data/raw/ and write them to data/processed/.

    python src/build_dataset.py

Outputs:
  traces.csv         one row per kept trace: architecture, failure origin, model, family, leaf
  dropped.csv        every removed trace and the reason
  leaves.csv         traces per leaf (and per dataset)
  leaf_families.csv  traces per leaf, model family and raw model name
  clusters/          the actual traces: <leaf>/<family>.jsonl, plus images/ (see export.py)
"""
from pathlib import Path

import pandas as pd
import yaml

import export
import taxonomy
import validate
from loaders import mascope, whowhenpro

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs" / "taxonomy_rules.yaml", encoding="utf-8"))
raw, out = ROOT / cfg["paths"]["raw"], ROOT / cfg["paths"]["processed"]

df = pd.concat([mascope.load(raw), whowhenpro.load(raw)], ignore_index=True)
df = taxonomy.assign(df, cfg)
validate.check(df, cfg)

kept = df[df.drop_reason.isna()]
out.mkdir(parents=True, exist_ok=True)
kept[["dataset", "corpus", "trace_id", "framework", "benchmark", "model", "model_family",
      "architecture", "failure_origin", "leaf"]].to_csv(out / "traces.csv", index=False)
df[df.drop_reason.notna()][["dataset", "corpus", "trace_id", "framework", "model", "failure_origin",
                            "trajectory_hash", "drop_reason"]].to_csv(out / "dropped.csv", index=False)
(kept.groupby(["leaf", "architecture", "failure_origin", "dataset"]).size().unstack(fill_value=0)
     .assign(traces=lambda t: t.sum(axis=1)).reset_index().to_csv(out / "leaves.csv", index=False))
(kept.groupby(["leaf", "model_family", "model"]).size().rename("traces").reset_index()
     .to_csv(out / "leaf_families.csv", index=False))

image_dir = raw / "whowhenpro" / cfg["sources"]["whowhenpro"]["pixelcraft_extract"]["images"]
export.write_clusters(kept, raw, image_dir, out / "clusters")
validate.check_clusters(kept, out / "clusters")

validate.summary(df)
print(f"\nwritten to {out.relative_to(ROOT)}/")
