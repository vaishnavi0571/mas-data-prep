"""MAScope -> common schema.

Dataset quirks handled here:
  * MAST-clean: original MAST runs, nothing injected; failures are the MAST annotations.
  * LG-Ctrl: an explicit anomaly_injected flag.
  * Inject-Mix: no flag; anomaly_type is set exactly when an anomaly was injected.
  * Every trace carries its own topology label (guo_topology).
"""
import hashlib
import json

import pandas as pd

from loaders import COLUMNS

CORPORA = ["MAST-clean", "LG-Ctrl", "Inject-Mix"]


def _injected(corpus, r):
    if corpus == "MAST-clean":
        return False
    if corpus == "LG-Ctrl":
        return bool(r["anomaly_injected"])
    return r["anomaly_type"] is not None


def load(raw_dir):
    rows = []
    for corpus in CORPORA:
        with open(raw_dir / "mascope" / f"{corpus}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.append({
                    "dataset": "MAScope",
                    "corpus": corpus,
                    "trace_id": r["trace_id"],
                    "framework": r["framework"],
                    "benchmark": r["benchmark_name"],
                    "model": r["llm_name"],
                    "architecture_label": r["guo_topology"],
                    "injected": _injected(corpus, r),
                    "has_failure_label": any(r["mast_annotation"].values()),
                    "trajectory_hash": hashlib.sha1(r["trajectory"].encode("utf-8")).hexdigest(),
                })
    return pd.DataFrame(rows, columns=COLUMNS)
