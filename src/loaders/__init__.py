"""Each loader turns one dataset's raw files into the common schema below (one row per trace).

    dataset             "MAScope" | "WhoWhenPro"
    corpus              sub-collection inside the dataset (MAST-clean, LG-Ctrl, ... / text, video, image)
    trace_id            the dataset's own id
    framework           agent framework that produced the trace
    benchmark           task benchmark
    model               raw model name as released (None if not recorded)
    architecture_label  the dataset's own architecture label (None if it has none)
    injected            True if the failure was injected on purpose
    has_failure_label   True if the trace has at least one failure label (MAScope: any MAST code
                        set; Who&When Pro: always, since every trace has an injected error)
    trajectory_hash     SHA-1 of the trajectory, for finding duplicates (MAScope: its text;
                        Who&When Pro: its JSON with sorted keys)

Loaders only read and rename. They make no decisions; those are in taxonomy.py.
"""
COLUMNS = ["dataset", "corpus", "trace_id", "framework", "benchmark", "model",
           "architecture_label", "injected", "has_failure_label", "trajectory_hash"]
