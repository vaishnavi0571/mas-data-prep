"""Download MAScope and Who&When Pro into data/raw/, at the revisions pinned in configs/taxonomy_rules.yaml.

Hugging Face publishes a SHA-256 only for large files. A file that is already present is skipped
if it matches that SHA-256, or if it has none. The row count of every .jsonl file is then checked.
Run from the repo root:  python src/download.py
"""
import base64
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs" / "taxonomy_rules.yaml", encoding="utf-8"))
RAW = ROOT / CFG["paths"]["raw"]
HF = "https://huggingface.co"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def remote_hashes(repo, revision, files):
    """SHA-256 of the large (LFS) files we need, as listed by Hugging Face at this revision."""
    out = {}
    for folder in sorted({str(Path(f).parent).replace(".", "") for f in files}):
        url = f"{HF}/api/datasets/{repo}/tree/{revision}/{folder}"
        for item in json.load(urllib.request.urlopen(url)):
            if item.get("lfs"):
                out[item["path"]] = item["lfs"]["oid"]
    return out


def open_at(url, offset=0):
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)


def download(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open_at(url) as resp, open(tmp, "wb") as f:
        while block := resp.read(1 << 20):
            f.write(block)
    tmp.replace(dest)


def count_rows(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


# ------------------------------------------------------------------ PixelCraft extract
IMAGE_TYPES = {b"\x89PNG": "png", b"\xff\xd8\xff": "jpg", b"GIF8": "gif", b"RIFF": "webp"}


def save_image(data_b64, fingerprint, image_dir):
    """Write one embedded image to <image_dir>/<fingerprint>.<ext>, once."""
    if any(image_dir.glob(f"{fingerprint}.*")):
        return
    raw = base64.b64decode(data_b64)
    ext = next((e for magic, e in IMAGE_TYPES.items() if raw.startswith(magic)), "bin")
    (image_dir / f"{fingerprint}.{ext}").write_bytes(raw)


def strip_images(x, image_dir):
    """Replace every embedded image with its fingerprint (and save the image as a file);
    leave everything else untouched.

    An embedded image is {"__type__": "PIL.Image", "data": <base64 text>}. It becomes
    {"__image__": <fingerprint>, "base64_chars": <length of data>}; any other key in the image
    object is dropped. The PixelCraft rows hold 8,068 image references: the 2,358 inside `task`
    were checked during exploration and have no other keys; the 5,710 inside `trajectory` were
    not checked.
    The fingerprint is the SHA-1 of the base64 text as released, not of the decoded PNG file,
    so hashing a saved PNG does not reproduce its name."""
    if isinstance(x, dict):
        if x.get("__type__") == "PIL.Image" and isinstance(x.get("data"), str):
            fingerprint = hashlib.sha1(x["data"].encode()).hexdigest()
            save_image(x["data"], fingerprint, image_dir)
            return {"__image__": fingerprint, "base64_chars": len(x["data"])}
        return {k: strip_images(v, image_dir) for k, v in x.items()}
    if isinstance(x, list):
        return [strip_images(v, image_dir) for v in x]
    return x


def parse(v):
    if isinstance(v, str) and v[:1] in "{[":
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def extract_pixelcraft(url, dest, image_dir, expected_sha256):
    """Stream the 6.8 GB image split line by line (resuming if the connection drops).
    Keeps the PixelCraft rows as text, saves their images as files, and checks that the
    streamed bytes have Hugging Face's SHA-256 for the pinned file."""
    resp = open_at(url)
    total = int(resp.headers["Content-Length"])
    offset = kept = retries = 0
    source_hash = hashlib.sha256()
    image_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with open(tmp, "w", encoding="utf-8") as out:
        while offset < total:
            try:
                line = resp.readline()
                if not line or (not line.endswith(b"\n") and offset + len(line) < total):
                    raise IOError("connection closed early")
            except Exception as e:
                retries += 1
                if retries > 50:
                    sys.exit(f"giving up at byte {offset}/{total}: {e}")
                time.sleep(5)
                resp = open_at(url, offset)
                continue
            offset += len(line)
            source_hash.update(line)
            row = json.loads(line)
            if row.get("framework") == "pixelcraft":
                row = {k: strip_images(parse(v), image_dir) for k, v in row.items()}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
    assert offset == total, "did not read the whole file"
    if source_hash.hexdigest() != expected_sha256:
        sys.exit("  streamed image.jsonl does not match Hugging Face's SHA-256")
    tmp.replace(dest)
    n_images = sum(1 for _ in image_dir.iterdir())
    print(f"  {dest.name}: {kept} rows, {n_images} images; source SHA-256 matches Hugging Face")


# ------------------------------------------------------------------ main
for name, src in CFG["sources"].items():
    out_dir = RAW / name
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_hash = remote_hashes(src["repo"], src["revision"], src["files"])
    print(f"{name}  ({src['repo']} @ {src['revision'][:8]})")

    for file in src["files"]:
        dest = out_dir / Path(file).name
        url = f"{HF}/datasets/{src['repo']}/resolve/{src['revision']}/{file}"
        want = expected_hash.get(file)
        if dest.exists() and (want is None or sha256(dest) == want):
            print(f"  {dest.name}: present")
            continue
        print(f"  {dest.name}: downloading")
        download(url, dest)
        if want and sha256(dest) != want:
            sys.exit(f"  {dest.name}: SHA-256 does not match Hugging Face")

    ex = src.get("pixelcraft_extract")
    if ex:
        dest, image_dir = out_dir / ex["to"], out_dir / ex["images"]
        if dest.exists() and image_dir.exists():
            print(f"  {dest.name} + {image_dir.name}/: present")
        else:
            print(f"  {dest.name} + {image_dir.name}/: streaming {ex['from']} (6.8 GB, about an hour)")
            url = f"{HF}/datasets/{src['repo']}/resolve/{src['revision']}/{ex['from']}"
            extract_pixelcraft(url, dest, image_dir, expected_hash[ex["from"]])

    for file, n in src["expected_rows"].items():
        got = count_rows(out_dir / file)
        assert got == n, f"{file}: {got} rows, expected {n}"
    print(f"  row counts OK: {src['expected_rows']}")
