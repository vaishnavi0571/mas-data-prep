"""Turn the common-schema table into leaves. Every rule comes from the config.

    assign(df, cfg) adds:
        architecture    dataset label if there is one, else the config's framework map
        failure_origin  synthetic | natural | no_failure
        drop_reason     None for kept traces, else why the trace was removed
        model_family    from the config's regexes
        leaf            "<architecture> / <failure_origin>"
"""
import re


def failure_origin(df):
    origin = df.has_failure_label.map({True: "natural", False: "no_failure"})
    return origin.mask(df.injected, "synthetic")


def model_family(model, rules):
    if model is None:
        return None
    for rule in rules:
        if re.search(rule["pattern"], model, re.IGNORECASE):
            return rule["family"]
    return None


def assign(df, cfg):
    df = df.copy()
    df["architecture"] = df.architecture_label.fillna(df.framework.map(cfg["architecture_by_framework"]))
    df["failure_origin"] = failure_origin(df)
    df["drop_reason"] = None

    def drop(mask, reason):
        df.loc[mask & df.drop_reason.isna(), "drop_reason"] = reason

    # 1. duplicates: same trajectory within a dataset
    if cfg["dedup"]:
        key = ["dataset", "trajectory_hash"]
        g = df.groupby(key)
        is_dup = g.trace_id.transform("size") > 1
        agree = (g.failure_origin.transform("nunique") == 1) & \
                (g.architecture.transform(lambda s: s.astype(str).nunique()) == 1)
        drop(is_dup & ~agree, "duplicate_conflicting")               # labels disagree: drop every copy
        drop(is_dup & agree & df.duplicated(key), "duplicate_copy")  # labels agree: keep the first

    # 2. model filter
    if cfg["require_model"]:
        drop(df.model.isna(), "no_model")

    # 3. failure filter
    if cfg["drop_no_failure"]:
        drop(df.failure_origin == "no_failure", "no_failure")

    df["model_family"] = df.model.map(lambda m: model_family(m, cfg["model_families"]))
    df["leaf"] = df.architecture + " / " + df.failure_origin
    return df
