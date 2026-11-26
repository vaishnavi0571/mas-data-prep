"""Who&When Pro -> common schema.

Dataset quirks handled here:
  * task / trajectory / ground_truth / extras are stored as JSON strings.
  * The model, when recorded, is in extras.agents[].model. Only PixelCraft records it,
    and all agents in a trace share the same model.
  * Every trace has exactly one injected error (the benchmark's design).
  * No architecture label is given; it is mapped per framework in the config.
"""
import hashlib
import json

import pandas as pd

from loaders import COLUMNS

FILES = {"text": "text.jsonl", "video": "video.jsonl", "image": "pixelcraft.jsonl"}


def _parse(v):
    return json.loads(v) if isinstance(v, str) and v[:1] in "{[" else v


def load(raw_dir):
    rows = []
    for corpus, file in FILES.items():
        with open(raw_dir / "whowhenpro" / file, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                agents = (_parse(r["extras"]) or {}).get("agents", [])
                models = sorted({a["model"] for a in agents if a.get("model")})
                trajectory = json.dumps(_parse(r["trajectory"]), sort_keys=True, ensure_ascii=False)
                rows.append({
                    "dataset": "WhoWhenPro",
                    "corpus": corpus,
                    "trace_id": r["id"],
                    "framework": r["framework"],
                    "benchmark": r["benchmark"],
                    "model": "+".join(models) or None,
                    "architecture_label": None,
                    "injected": True,
                    "has_failure_label": True,
                    "trajectory_hash": hashlib.sha1(trajectory.encode("utf-8")).hexdigest(),
                })
    return pd.DataFrame(rows, columns=COLUMNS)
