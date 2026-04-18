"""
evaluate.py
===========
Benchmark Scorer for Phases 2–5

WHAT THIS FILE DOES
-------------------
This file is the measurement layer of the pipeline. It does not detect
anything — detection is the job of Phases 2–5. This file answers one
question: "Given a set of indices your detector flagged, how well did
it actually do against the ground truth?"

It provides three things:
    1. evaluate_flags()     — precision, recall, F1 for a single threshold
    2. build_pr_curve()     — sweeps a list of thresholds, returns full PR data
    3. summarize_manifest() — prints a human-readable summary of ground_truth.json

WORKFLOW
--------
This file is NOT called during Phase 0.
Phase 0 only writes ground_truth.json and exits.

evaluate.py is called by Phases 2–5, after each phase produces a set of
flagged image indices. The typical call pattern per phase is:

    Step 1 — Run your detector at an initial threshold.
             Collect the flagged indices as a Python set or list.

    Step 2 — Call evaluate_flags() with those indices and the corruption
             type(s) you are targeting. Read precision, recall, F1.

    Step 3 — Repeat Step 1–2 across a range of thresholds (e.g. sweep ε
             for Phase 2, or percentile cutoffs for Phase 3).

    Step 4 — Call build_pr_curve() with all threshold results collected
             in Step 3. It returns the full curve data.

    Step 5 — Select the threshold at maximum F1. Record it in README.md.

    Step 6 — Save your flagged indices CSV with the chosen threshold.

HOW PRECISION AND RECALL WORK HERE
------------------------------------
    True Positive (TP)  : flagged by detector AND present in ground truth
    False Positive (FP) : flagged by detector but NOT in ground truth (wrong flag)
    False Negative (FN) : in ground truth but NOT flagged by detector (missed)

    Precision = TP / (TP + FP)   — of everything we flagged, how much was real?
    Recall    = TP / (TP + FN)   — of everything real, how much did we catch?
    F1        = 2 * P * R / (P + R)  — harmonic mean; balances the two

Each phase targets a specific corruption type:
    Phase 2 → "duplicate"
    Phase 3 → "image_corruption"
    Phase 4 → "label_noise"
    Phase 5 → uses Phase 3 + Phase 4 results combined; no new evaluate call

Run via:
    from src.benchmark.evaluate import evaluate_flags, build_pr_curve, summarize_manifest
Never run this file directly.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union

# ─────────────────────────────────────────────────────────────────────────────
# PATH CONSTANT
# ─────────────────────────────────────────────────────────────────────────────

GROUND_TRUTH_PATH = "data/04_benchmark/ground_truth.json"

# Valid corruption type keys in ground_truth.json.
# Passing anything outside this set is a programming error, not a runtime one.
VALID_CORRUPTION_TYPES = {"label_noise", "image_corruption", "duplicate"}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: LOAD GROUND TRUTH
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth(path=GROUND_TRUTH_PATH):
    """
    Load and return ground_truth.json as a Python dict.

    Raises:
        FileNotFoundError if the file doesn't exist yet (Phase 0 not run).
        ValueError if the file is missing required keys.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ground_truth.json not found at '{path}'.\n"
            f"Run Phase 0 first: python -m src.main --mode benchmark"
        )

    with open(path, "r") as f:
        manifest = json.load(f)

    # Basic integrity check — all three corruption type keys must be present.
    for key in VALID_CORRUPTION_TYPES:
        if key not in manifest:
            raise ValueError(
                f"ground_truth.json is missing key '{key}'. "
                f"The file may be incomplete or corrupt. Re-run Phase 0."
            )

    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: BUILD THE GROUND TRUTH INDEX SET
# Extract the set of image indices for one or more corruption types.
# ─────────────────────────────────────────────────────────────────────────────

def get_true_positive_set(manifest, corruption_types):
    """
    Given a manifest and a list of corruption type names, return the set of
    all image indices that were injected under those types.

    This becomes the "ground truth positive" set that detector output is
    compared against.

    Args:
        manifest         : dict loaded from ground_truth.json
        corruption_types : list of strings, e.g. ["image_corruption"] or
                           ["label_noise", "duplicate"]

    Returns:
        A set of integer image indices.

    Example:
        If ground_truth.json has label_noise entries at indices {5, 12, 88, ...},
        get_true_positive_set(manifest, ["label_noise"]) returns that set.
    """
    for ct in corruption_types:
        if ct not in VALID_CORRUPTION_TYPES:
            raise ValueError(
                f"Unknown corruption type '{ct}'. "
                f"Valid types are: {VALID_CORRUPTION_TYPES}"
            )

    true_set = set()
    for ct in corruption_types:
        for entry in manifest[ct]:
            true_set.add(entry["index"])

    return true_set


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTION 1: evaluate_flags()
# Score a single set of flagged indices against ground truth.
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_flags(
    flagged_indices: Union[set, list],
    corruption_types: list,
    ground_truth_path: str = GROUND_TRUTH_PATH,
    verbose: bool = True,
) -> dict:
    """
    Compute precision, recall, and F1 for a single detector output.

    This is the function you call after running your detector at one threshold.

    Args:
        flagged_indices   : set or list of integer indices your detector flagged.
                            These are row numbers in embeddings.npy.
        corruption_types  : list of corruption type strings to score against.
                            Use ["duplicate"] for Phase 2,
                                ["image_corruption"] for Phase 3,
                                ["label_noise"] for Phase 4.
        ground_truth_path : path to ground_truth.json. Default is the standard path.
        verbose           : if True, prints a formatted result table.

    Returns:
        A dict with keys:
            "precision"      : float in [0, 1]
            "recall"         : float in [0, 1]
            "f1"             : float in [0, 1]
            "tp"             : int — true positives
            "fp"             : int — false positives (detector was wrong)
            "fn"             : int — false negatives (detector missed these)
            "n_flagged"      : int — total flags raised by detector
            "n_ground_truth" : int — total real corruptions of the target type(s)

    Edge cases:
        If no flags are raised → precision is undefined, set to 0.0.
        If ground truth is empty → recall is undefined, set to 0.0.
        F1 is 0.0 if either precision or recall is 0.0.

    Typical call (Phase 3 example):
        results = evaluate_flags(
            flagged_indices  = outlier_indices,
            corruption_types = ["image_corruption"],
        )
        print(results["f1"])
    """
    manifest = load_ground_truth(ground_truth_path)
    flagged_set = set(flagged_indices)
    true_set    = get_true_positive_set(manifest, corruption_types)

    # ── COMPUTE TP, FP, FN ────────────────────────────────────────────────────
    # Set intersection / difference does all the heavy lifting.
    tp = len(flagged_set & true_set)    # flagged AND in ground truth
    fp = len(flagged_set - true_set)    # flagged but NOT in ground truth
    fn = len(true_set - flagged_set)    # in ground truth but NOT flagged

    # ── COMPUTE METRICS ────────────────────────────────────────────────────────
    # Guard against division by zero in edge cases.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) \
                if (precision + recall) > 0 else 0.0

    results = {
        "precision":      round(precision, 4),
        "recall":         round(recall, 4),
        "f1":             round(f1, 4),
        "tp":             tp,
        "fp":             fp,
        "fn":             fn,
        "n_flagged":      len(flagged_set),
        "n_ground_truth": len(true_set),
    }

    if verbose:
        _print_evaluation_table(results, corruption_types)

    return results


def _print_evaluation_table(results, corruption_types):
    """
    Internal helper. Prints a formatted table of evaluation results.
    Not intended to be called directly.
    """
    print("\n" + "-" * 50)
    print(f"  Evaluation Results  |  Target: {corruption_types}")
    print("-" * 50)
    print(f"  Precision      : {results['precision']:.4f}")
    print(f"  Recall         : {results['recall']:.4f}")
    print(f"  F1 Score       : {results['f1']:.4f}")
    print("─" * 50)
    print(f"  True Positives : {results['tp']}")
    print(f"  False Positives: {results['fp']}  <-- detector was wrong")
    print(f"  False Negatives: {results['fn']}  <-- detector missed these")
    print("-" * 50)
    print(f"  Flagged total  : {results['n_flagged']}")
    print(f"  Ground truth   : {results['n_ground_truth']}")
    print("-" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTION 2: build_pr_curve()
# Sweep thresholds and collect precision/recall at each step.
# ─────────────────────────────────────────────────────────────────────────────

def build_pr_curve(
    threshold_flag_pairs: list,
    corruption_types: list,
    ground_truth_path: str = GROUND_TRUTH_PATH,
    plot: bool = True,
    save_path: Optional[str] = None,
    threshold_label: str = "Threshold",
) -> dict:
    """
    Build a precision-recall curve by evaluating the detector at multiple
    thresholds and collecting the results.

    You call this after you've already run your detector at many thresholds
    and collected (threshold, flagged_indices) pairs.

    Args:
        threshold_flag_pairs : list of (threshold_value, flagged_indices) tuples.
                               threshold_value can be any float (ε, percentile, etc.)
                               flagged_indices is a set/list of ints at that threshold.

                               Example for Phase 2 (epsilon sweep):
                                   [
                                       (0.01, {5, 12, 88}),
                                       (0.02, {5, 12, 44, 88, 101}),
                                       (0.05, {5, 12, 44, 67, 88, 101, 200}),
                                       ...
                                   ]

        corruption_types     : same as evaluate_flags() — which types to score against.
        ground_truth_path    : path to ground_truth.json.
        plot                 : if True, renders a matplotlib PR curve inline.
        save_path            : if provided, saves the plot to this path as a PNG.
                               e.g. "data/03_reports/phase2_pr_curve.png"

    Returns:
        A dict with keys:
            "thresholds"  : list of threshold values (same order as input)
            "precisions"  : list of precision values, one per threshold
            "recalls"     : list of recall values, one per threshold
            "f1s"         : list of F1 values, one per threshold
            "best_threshold"     : threshold value at maximum F1
            "best_f1"            : F1 value at the best threshold
            "best_precision"     : precision at the best threshold
            "best_recall"        : recall at the best threshold
            "best_threshold_idx" : index into the lists where best F1 occurs

    Workflow note:
        After calling this, look at best_threshold and record it in README.md.
        Then re-run your detector with ONLY that threshold to produce the final
        flagged CSV for this phase.
    """
    manifest    = load_ground_truth(ground_truth_path)
    true_set    = get_true_positive_set(manifest, corruption_types)

    thresholds  = []
    precisions  = []
    recalls     = []
    f1s         = []

    print(f"\n[PR Curve] Sweeping {len(threshold_flag_pairs)} thresholds "
          f"against '{corruption_types}' ...")

    for threshold, flagged_indices in threshold_flag_pairs:
        # evaluate_flags with verbose=False to suppress per-step printing.
        # We only want the summary at the end.
        result = evaluate_flags(
            flagged_indices   = flagged_indices,
            corruption_types  = corruption_types,
            ground_truth_path = ground_truth_path,
            verbose           = False,
        )
        thresholds.append(threshold)
        precisions.append(result["precision"])
        recalls.append(result["recall"])
        f1s.append(result["f1"])

    # ── FIND BEST THRESHOLD ────────────────────────────────────────────────────
    # argmax gives the index of the highest F1 in the list.
    best_idx       = int(np.argmax(f1s))
    best_threshold = thresholds[best_idx]
    best_f1        = f1s[best_idx]
    best_precision = precisions[best_idx]
    best_recall    = recalls[best_idx]

    print(f"\n[PR Curve] Best threshold: {best_threshold}")
    print(f"[PR Curve] Best F1: {best_f1:.4f}  "
          f"(Precision={best_precision:.4f}, Recall={best_recall:.4f})")

    # ── PLOT ──────────────────────────────────────────────────────────────────
    if plot:
        _plot_pr_curve(
            recalls, precisions, f1s, thresholds,
            best_idx, corruption_types, save_path, threshold_label
        )

    return {
        "thresholds":         thresholds,
        "precisions":         precisions,
        "recalls":            recalls,
        "f1s":                f1s,
        "best_threshold":     best_threshold,
        "best_f1":            best_f1,
        "best_precision":     best_precision,
        "best_recall":        best_recall,
        "best_threshold_idx": best_idx,
    }


def _plot_pr_curve(recalls, precisions, f1s, thresholds,
                   best_idx, corruption_types, save_path,
                   threshold_label="Threshold"):
    """
    Internal helper. Renders the PR curve and optionally saves it.

    Plots two panels:
        Left  — Precision vs Recall curve (the standard PR curve).
                The best-threshold point is highlighted in red.
        Right — F1 vs Threshold line chart.
                Useful for understanding where the detector peaks.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    title_str = f"PR Curve -- {corruption_types}"

    # ── LEFT PANEL: Precision vs Recall ───────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(recalls, precisions, marker="o", markersize=4,
             linewidth=1.5, color="steelblue", label="P-R curve")

    # Highlight the best-F1 operating point.
    ax1.scatter(
        recalls[best_idx], precisions[best_idx],
        color="red", zorder=5, s=80,
        label=f"Best F1={f1s[best_idx]:.3f} @ threshold={thresholds[best_idx]}"
    )

    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title(f"Precision–Recall Curve\n{title_str}")
    ax1.set_xlim([0, 1.05])
    ax1.set_ylim([0, 1.05])
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── RIGHT PANEL: F1 vs Threshold ──────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(thresholds, f1s, marker="o", markersize=4,
             linewidth=1.5, color="darkorange", label="F1")
    ax2.axvline(
        x=thresholds[best_idx], color="red", linestyle="--", linewidth=1.2,
        label=f"Best threshold={thresholds[best_idx]}"
    )
    ax2.set_xlabel(threshold_label)
    ax2.set_ylabel("F1 Score")
    ax2.set_title(f"F1 vs Threshold\n{title_str}")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[PR Curve] Plot saved to '{save_path}'.")

    # Only call plt.show() in interactive environments (notebooks, GUI).
    # In CLI pipeline mode, plt.show() blocks execution until the user
    # manually closes the window. Always close the figure to free memory.
    if plt.isinteractive() or plt.get_backend().lower().startswith("qt"):
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTION 3: summarize_manifest()
# Human-readable printout of what ground_truth.json contains.
# ─────────────────────────────────────────────────────────────────────────────

def summarize_manifest(ground_truth_path: str = GROUND_TRUTH_PATH):
    """
    Print a human-readable summary of ground_truth.json.

    Use this at the start of any phase to quickly confirm the benchmark
    you're scoring against is what you expect — correct seed, correct counts,
    correct corruption rates.

    Prints:
        - Seed used in Phase 0
        - Total image count
        - Per-corruption-type counts and rates
        - A few example entries from each type

    No return value. Pure side effect (print).

    Typical call:
        from src.benchmark.evaluate import summarize_manifest
        summarize_manifest()
    """
    manifest = load_ground_truth(ground_truth_path)

    N    = manifest.get("total_images", "unknown")
    seed = manifest.get("seed", "unknown")
    cfg  = manifest.get("corruption_config", {})

    print("\n" + "=" * 60)
    print("  GROUND TRUTH MANIFEST SUMMARY")
    print("=" * 60)
    print(f"  File            : {ground_truth_path}")
    print(f"  Total images    : {N}")
    print(f"  Random seed     : {seed}")
    print("-" * 60)

    for ctype in VALID_CORRUPTION_TYPES:
        entries = manifest.get(ctype, [])
        count   = len(entries)
        rate    = cfg.get(f"{ctype}_rate", "?")

        print(f"\n  [{ctype}]")
        print(f"    Count : {count}  (configured rate: {rate})")

        if count == 0:
            print("    (no entries)")
            continue

        # Show up to 3 example entries so you can visually spot-check the manifest.
        print(f"    Sample entries (up to 3):")
        for entry in entries[:3]:
            # Show only the most informative fields per type.
            if ctype == "label_noise":
                print(f"      idx={entry['index']}  "
                      f"{entry['original_label']} -> {entry['injected_label']}  "
                      f"| {os.path.basename(entry['image_path'])}")
            elif ctype == "image_corruption":
                print(f"      idx={entry['index']}  "
                      f"label={entry['original_label']}  "
                      f"| {os.path.basename(entry['image_path'])}")
            elif ctype == "duplicate":
                print(f"      idx={entry['index']}  "
                      f"original={os.path.basename(entry.get('original_path', '?'))}  "
                      f"-> dup={os.path.basename(entry.get('duplicate_path', '?'))}")

    print("\n" + "=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLES (reference — not executed on import)
# ─────────────────────────────────────────────────────────────────────────────

"""
─── PHASE 2 EXAMPLE (Near-Duplicate Detection) ───────────────────────────────

from src.benchmark.evaluate import evaluate_flags, build_pr_curve

# Single threshold evaluation:
results = evaluate_flags(
    flagged_indices  = my_duplicate_index_set,   # set of ints from find_duplicates.py
    corruption_types = ["duplicate"],
)

# Threshold sweep (ε from 0.01 to 0.20):
epsilon_values = [round(e * 0.01, 2) for e in range(1, 21)]
pairs = []
for eps in epsilon_values:
    flagged = run_duplicate_detector(embeddings, eps)   # your detector function
    pairs.append((eps, flagged))

curve = build_pr_curve(
    threshold_flag_pairs = pairs,
    corruption_types     = ["duplicate"],
    save_path            = "data/03_reports/phase2_pr_curve.png",
)
best_eps = curve["best_threshold"]
print(f"Use ε = {best_eps} for final duplicate detection.")


─── PHASE 3 EXAMPLE (Structural Outlier Detection) ──────────────────────────

results = evaluate_flags(
    flagged_indices  = outlier_index_set,
    corruption_types = ["image_corruption"],
)

# Percentile sweep (80th to 99th):
pairs = []
for pct in range(80, 100):
    flagged = flag_above_percentile(mahal_distances, pct)
    pairs.append((pct, flagged))

curve = build_pr_curve(pairs, ["image_corruption"],
                       save_path="data/03_reports/phase3_pr_curve.png")


─── PHASE 4 EXAMPLE (Label Error Detection) ─────────────────────────────────

results = evaluate_flags(
    flagged_indices  = cleanlab_flagged_set,
    corruption_types = ["label_noise"],
)


─── MANIFEST SUMMARY (any phase) ────────────────────────────────────────────

from src.benchmark.evaluate import summarize_manifest
summarize_manifest()
"""