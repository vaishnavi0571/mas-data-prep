# MAS data prep

This repo sorts agent traces from **MAScope** and **Who&When Pro** into leaves by
**agent architecture** and **failure origin** (natural vs. synthetic), then groups
each leaf's traces by **model family**.

## Reproduce

```bash
pip install -r requirements.txt
python src/download.py        # data/raw/: pinned revisions, checked by SHA-256 and row counts
python src/build_dataset.py   # data/processed/: dedup, filters, leaves, families, trace export, validation
```

`download.py` takes about an hour the first time, because it streams the 6.8 GB image split. Then open
`notebooks/02_audit_leaves.ipynb` for the summary tables and charts.

## Outputs (`data/processed/`)

| Output | Contents |
|---|---|
| `clusters/<leaf>/<family>.jsonl` | **the actual traces**, one folder per leaf and one file per model family (35 files). Each line holds our labels plus `trace`, the full original record |
| `clusters/images/` | PixelCraft chart images used by those traces (PNG, named by fingerprint; each image's `file` field points here) |
| `traces.csv` | index: one row per kept trace with its leaf, family and original `trace_id` |
| `dropped.csv` | every removed trace and why |
| `leaves.csv`, `leaf_families.csv` | counts per leaf, and per leaf × family × raw model name |

`trace_id` is the dataset's own ID, so every trace can be found in the original release.

## Layout

```
configs/taxonomy_rules.yaml   every decision with its reason: sources, architecture map, filters, family regexes
src/
  download.py                 → data/raw/ (never edited)
  loaders/mascope.py          raw MAScope       → common schema (see loaders/__init__.py)
  loaders/whowhenpro.py       raw Who&When Pro  → common schema
  taxonomy.py                 dedup, filters, architecture, failure origin, model family, leaf
  export.py                   writes the actual traces into clusters/<leaf>/<family>.jsonl
  validate.py                 hard checks; the build stops if any fail
  build_dataset.py            entry point; writes data/processed/
notebooks/
  01_eda.ipynb                exploration of both datasets
  02_audit_leaves.ipynb       final tables and charts
```

`data/` is not committed. Running `download.py` rebuilds it.

## Rules

| Step | Rule |
|---|---|
| Architecture | MAScope: the trace's own `guo_topology` label. Who&When Pro: per framework in the config (PixelCraft → Centralized) |
| Failure origin | **synthetic**: the failure was injected. **natural**: not injected, but annotators found at least one failure. **no failure**: neither |
| Duplicates | identical trajectory text counts as one trace. If the copies' labels agree, keep one; if they disagree, drop every copy |
| Model filter | keep only traces that name their model (in Who&When Pro, only PixelCraft does) |
| Failure filter | drop traces with no failure |
| Model family | regex on the raw model name (config). The raw name is kept alongside |

## Result

| | Traces |
|---|---:|
| Loaded | 11,795 |
| Dropped: duplicate with conflicting labels | 11 |
| Dropped: duplicate copy | 6 |
| Dropped: no model name | 8,071 |
| Dropped: no failure | 332 |
| **Kept** | **3,375** (MAScope 1,871 + Who&When Pro 1,504) |

| Leaf | Traces | Families |
|---|---:|---|
| Centralized / synthetic | 1,811 | Gemini 1,576 (1,504 PixelCraft), GPT, MiniMax, Claude, DeepSeek, Doubao, GLM |
| Decentralized / synthetic | 505 | Gemini, GPT, MiniMax, Claude, GLM, DeepSeek, Doubao |
| Shared Message Pool / natural | 408 | GPT, Llama, Claude, Qwen |
| Layered / natural | 235 | GPT, Qwen, Llama |
| Layered / synthetic | 215 | Gemini, GPT, GLM, DeepSeek, Claude, Doubao, MiniMax |
| Centralized / natural | 130 | GPT |
| Shared Message Pool / synthetic | 71 | Gemini, GPT, MiniMax, GLM, DeepSeek, Claude |

Every removed trace is listed in `data/processed/dropped.csv` with its reason.

## Sources

- MAScope: [`mascope/mascope-bench`](https://huggingface.co/datasets/mascope/mascope-bench)
- Who&When Pro: [`Leoxx/whowhen_pro`](https://huggingface.co/datasets/Leoxx/whowhen_pro), the official release.
  The image split (6.8 GB) is streamed once, and its SHA-256 is checked against Hugging Face's.
  Only the PixelCraft rows are kept: the text unchanged, except that each embedded image is saved
  as a PNG in `data/raw/whowhenpro/pixelcraft_images/` and the text points to it by file name
  (its fingerprint), as `{"__image__": <fingerprint>, "base64_chars": <length of the base64 text>}`.
  The fingerprint is the SHA-1 of the image's base64 text as released, not of the PNG file.
