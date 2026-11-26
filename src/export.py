"""Write the actual traces into one folder per leaf and one file per model family:

    data/processed/clusters/
        <Architecture>__<origin>/<Family>.jsonl
        images/<fingerprint>.<ext>        PixelCraft images used by the kept traces, stored once

Each line is one trace: our labels, plus "trace" = the original record from data/raw/.
  * MAScope records are copied as they are.
  * Who&When Pro records have their JSON-string fields (task, trajectory, ground_truth, extras)
    parsed into objects, and every image fingerprint gets a "file" path (relative to clusters/).
    An image then reads {"__image__": <fingerprint>, "base64_chars": n, "file": "images/<fingerprint>.png"};
    the fingerprint is the SHA-1 of the released base64 text (see download.py).
"""
import json
import shutil

LABELS = ["trace_id", "dataset", "corpus", "framework", "benchmark", "model", "model_family",
          "architecture", "failure_origin", "leaf"]


def folder_name(leaf):
    architecture, origin = leaf.split(" / ")
    return f"{architecture.replace(' ', '_')}__{origin}"


def _parse(v):
    return json.loads(v) if isinstance(v, str) and v[:1] in "{[" else v


def _link_images(x, files, used):
    if isinstance(x, dict):
        if "__image__" in x:
            name = files[x["__image__"]]
            used.add(name)
            return {**x, "file": f"images/{name}"}
        return {k: _link_images(v, files, used) for k, v in x.items()}
    if isinstance(x, list):
        return [_link_images(v, files, used) for v in x]
    return x


def _raw_records(raw, kept_ids, image_files, used_images):
    """Yield (trace_id, original record) for every kept trace."""
    for corpus in ["MAST-clean", "LG-Ctrl", "Inject-Mix"]:
        with open(raw / "mascope" / f"{corpus}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["trace_id"] in kept_ids:
                    yield r["trace_id"], r
    for file in ["text.jsonl", "video.jsonl", "pixelcraft.jsonl"]:
        with open(raw / "whowhenpro" / file, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["id"] in kept_ids:
                    r = {k: _parse(v) for k, v in r.items()}
                    yield r["id"], _link_images(r, image_files, used_images)


def write_clusters(kept, raw, image_dir, out):
    if out.exists():
        shutil.rmtree(out)                      # never leave stale files from an earlier run
    (out / "images").mkdir(parents=True)

    image_files = {p.stem: p.name for p in image_dir.iterdir()}
    used_images = set()
    labels = kept.set_index("trace_id")[LABELS[1:]].to_dict("index")
    handles = {}
    try:
        for trace_id, record in _raw_records(raw, set(labels), image_files, used_images):
            lab = labels[trace_id]
            path = out / folder_name(lab["leaf"]) / f"{lab['model_family']}.jsonl"
            if path not in handles:
                path.parent.mkdir(exist_ok=True)
                handles[path] = open(path, "w", encoding="utf-8")
            handles[path].write(json.dumps({"trace_id": trace_id, **lab, "trace": record},
                                           ensure_ascii=False) + "\n")
    finally:
        for h in handles.values():
            h.close()

    for name in used_images:
        shutil.copy2(image_dir / name, out / "images" / name)
