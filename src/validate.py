"""Hard checks on the pipeline result. Any failure stops the build with a clear message."""
from loaders import COLUMNS

ARCHITECTURES = {"Layered", "Centralized", "Decentralized", "Shared Message Pool"}
ORIGINS = {"synthetic", "natural", "no_failure"}


def check(df, cfg):
    kept = df[df.drop_reason.isna()]
    problems = []

    def expect(ok, message):
        if not ok:
            problems.append(message)

    # loading
    expect(set(COLUMNS) <= set(df.columns), "loader output is missing schema columns")
    expected = {name: sum(src["expected_rows"].values()) for name, src in cfg["sources"].items()}
    loaded = df.dataset.str.lower().value_counts().to_dict()
    expect(loaded == expected, f"rows loaded {loaded}, config expects {expected}")
    expect(not df.duplicated(["dataset", "trace_id"]).any(), "duplicate trace_id within a dataset")
    expect(df.failure_origin.isin(ORIGINS).all(), "unknown failure_origin value")

    # bookkeeping: every trace is either kept or dropped with a reason
    expect(len(kept) + df.drop_reason.notna().sum() == len(df), "traces lost between load and output")

    # kept traces: nothing unmapped
    expect(kept.architecture.notna().all(), "kept trace with no architecture (add its framework to the config)")
    expect(kept.architecture.isin(ARCHITECTURES).all(), "kept trace with an unknown architecture")
    if cfg["require_model"]:
        expect(kept.model.notna().all(), "kept trace with no model")
    unmapped = sorted(kept.loc[kept.model_family.isna() & kept.model.notna(), "model"].unique())
    expect(not unmapped, f"model names with no family regex: {unmapped}")
    if cfg["drop_no_failure"]:
        expect((kept.failure_origin != "no_failure").all(), "kept trace with no failure")
    if cfg["dedup"]:
        expect(not kept.duplicated(["dataset", "trajectory_hash"]).any(), "duplicate trajectories survived dedup")
    expect(kept.leaf.notna().all() and kept.groupby("leaf").size().min() > 0, "empty or missing leaf")

    if problems:
        raise SystemExit("VALIDATION FAILED\n  - " + "\n  - ".join(problems))
    print("validation passed: all checks OK")


def check_clusters(kept, out):
    """Every kept trace is in exactly one cluster file, in the right one, and every image it uses exists."""
    import json
    seen, problems, n_images = {}, [], 0
    for path in out.glob("*/*.jsonl"):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            seen[r["trace_id"]] = seen.get(r["trace_id"], 0) + 1
            if path.parent.name != r["leaf"].replace(" / ", "__").replace(" ", "_") or path.stem != r["model_family"]:
                problems.append(f"{r['trace_id']} is in the wrong file {path.parent.name}/{path.name}")
            for f in _image_files(r["trace"]):
                n_images += 1
                if not (out / f).exists():
                    problems.append(f"{r['trace_id']}: missing {f}")
    if set(seen) != set(kept.trace_id):
        problems.append(f"{len(set(kept.trace_id) - set(seen))} kept traces missing, "
                        f"{len(set(seen) - set(kept.trace_id))} unexpected traces")
    problems += [f"{t} written {n} times" for t, n in seen.items() if n > 1]
    if problems:
        raise SystemExit("CLUSTER EXPORT FAILED\n  - " + "\n  - ".join(problems[:20]))
    print(f"clusters passed: {len(seen)} traces in {len(list(out.glob('*/*.jsonl')))} files, "
          f"{n_images} image references all resolve")


def _image_files(x):
    if isinstance(x, dict):
        if "file" in x and "__image__" in x:
            yield x["file"]
        for v in x.values():
            yield from _image_files(v)
    elif isinstance(x, list):
        for v in x:
            yield from _image_files(v)


def summary(df):
    kept = df[df.drop_reason.isna()]
    print(f"\nloaded  {len(df):>6}")
    for reason, n in df.drop_reason.value_counts().items():
        print(f"dropped {n:>6}  {reason}")
    print(f"kept    {len(kept):>6}\n")
    table = kept.pivot_table(index="leaf", columns="model_family", values="trace_id",
                             aggfunc="count", fill_value=0, margins=True, margins_name="total")
    print(table.to_string())
