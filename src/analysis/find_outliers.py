"""
find_outliers.py
================
Phase 3: Structural Outlier Detection (Mahalanobis Distance)

WHAT THIS FILE DOES
-------------------
Identifies structurally anomalous images — corrupted files, images from the
wrong domain, or broken sensors — by computing the Mahalanobis distance of
every embedding from the distribution centroid.

The Mahalanobis distance accounts for the shape of the embedding distribution
(via the covariance matrix), so "far away" means "statistically unusual"
rather than "large in some arbitrary dimension". This is critical because
embedding dimensions are NOT equally informative.

Images at the extreme statistical tail of the distance distribution are
flagged as structural outliers. The tail cutoff is a percentile threshold:
    - percentile=95 → flag the top 5%
    - percentile=99 → flag the top 1%

In benchmark mode, the percentile is swept from 80 to 99, and the best
threshold is selected at maximum F1 against ground_truth.json["image_corruption"].

OUTPUTS
-------
    data/03_reports/outliers.csv
        Columns: index, mahalanobis_distance, path

Run via:
    python -m src.main --mode benchmark --phase 3
Never run this file directly.
"""

import os
import csv
import json
import numpy as np

from src.core_math.distributions import (
    compute_centroid,
    compute_covariance,
    invert_covariance,
)
from src.core_math.distances import mahalanobis_distances, flag_above_percentile


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

OUTLIER_CONFIG = {
    # Default percentile threshold for single-threshold detection.
    # Images with Mahalanobis distance above this percentile are flagged.
    "default_percentile":  95,

    # Percentile sweep range for PR curve construction.
    # 80 to 99 inclusive → 20 thresholds.
    "sweep_start":         80,
    "sweep_stop":          100,  # exclusive, so range is 80..99

    # Covariance regularisation (passed to compute_covariance).
    "regularisation":      1e-6,
}

# Path constants.
EMBEDDINGS_PATH = "data/02_embeddings/embeddings.npy"
INDEX_PATH      = "data/02_embeddings/index.json"
REPORTS_DIR     = "data/03_reports"
OUTPUT_CSV      = os.path.join(REPORTS_DIR, "outliers.csv")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: LOAD INPUTS
# ─────────────────────────────────────────────────────────────────────────────

def load_inputs():
    """
    Load the embedding matrix and image index from disk.

    Returns:
        embeddings  : np.ndarray of shape (N, 768), dtype float32.
        image_paths : list of N path strings.
    """
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"embeddings.npy not found at '{EMBEDDINGS_PATH}'. "
            f"Run Phase 1 first."
        )

    embeddings = np.load(EMBEDDINGS_PATH)

    # index.json is needed for path resolution in the CSV output.
    image_paths = []
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "r") as f:
            image_paths = json.load(f)

    print(f"[Phase 3] Loaded embeddings: {embeddings.shape}")
    print(f"[Phase 3] Loaded index: {len(image_paths)} entries")

    return embeddings, image_paths


# ─────────────────────────────────────────────────────────────────────────────
# CORE: COMPUTE MAHALANOBIS DISTANCES
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_distances(embeddings, config=None):
    """
    Compute the Mahalanobis distance of every embedding from the centroid.

    Steps:
        1. Compute centroid μ (mean of all embeddings).
        2. Compute covariance matrix Σ (with regularisation).
        3. Invert covariance → Σ⁻¹.
        4. Compute Mahalanobis distances for all rows.

    Args:
        embeddings : (N, 768) float32 array.
        config     : optional config dict.

    Returns:
        distances : np.ndarray of shape (N,), Mahalanobis distances.
    """
    if config is None:
        config = OUTLIER_CONFIG

    # Step 1: Centroid.
    print("\n[Phase 3] Step 1: Computing centroid ...")
    centroid = compute_centroid(embeddings)

    # Step 2: Covariance matrix.
    print("\n[Phase 3] Step 2: Computing covariance matrix ...")
    cov_matrix = compute_covariance(
        embeddings,
        regularisation=config["regularisation"],
    )

    # Step 3: Invert covariance.
    print("\n[Phase 3] Step 3: Inverting covariance matrix ...")
    cov_inv = invert_covariance(cov_matrix)

    # Step 4: Mahalanobis distances.
    print("\n[Phase 3] Step 4: Computing Mahalanobis distances ...")
    distances = mahalanobis_distances(embeddings, centroid, cov_inv)

    # Print distribution summary.
    print(f"\n[Phase 3] Distance statistics:")
    print(f"  Min    : {distances.min():.4f}")
    print(f"  Max    : {distances.max():.4f}")
    print(f"  Mean   : {distances.mean():.4f}")
    print(f"  Median : {np.median(distances):.4f}")
    print(f"  Std    : {distances.std():.4f}")

    return distances


# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS TO CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_outliers_csv(flagged_indices, distances, image_paths,
                      output_path=OUTPUT_CSV):
    """
    Write the flagged outlier indices to a CSV file.

    Columns: index, mahalanobis_distance, path

    Args:
        flagged_indices : set or list of flagged integer indices.
        distances       : (N,) array of Mahalanobis distances.
        image_paths     : list of image path strings.
        output_path     : CSV output file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sort flagged indices by distance descending (most outlier-like first).
    sorted_indices = sorted(flagged_indices, key=lambda i: distances[i], reverse=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "mahalanobis_distance", "path"])

        for idx in sorted_indices:
            path = image_paths[idx] if idx < len(image_paths) else "?"
            writer.writerow([idx, f"{distances[idx]:.6f}", path])

    print(f"\n[Save] outliers.csv -> '{output_path}' ({len(sorted_indices)} outliers)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_outlier_detection(benchmark_mode=False):
    """
    Orchestrates Phase 3: Structural Outlier Detection.

    Steps:
        1. Load embeddings.
        2. Compute Mahalanobis distances for all rows.
        3. In benchmark mode: sweep percentile thresholds, build PR curve,
           pick best threshold.
           In production mode: use default percentile.
        4. Flag indices above the chosen percentile.
        5. Save outliers.csv.

    Args:
        benchmark_mode : bool. If True, sweep thresholds and score against
                         ground_truth.json["image_corruption"].
    """
    config = OUTLIER_CONFIG.copy()
    embeddings, image_paths = load_inputs()

    # Compute Mahalanobis distances once — reused across all threshold sweeps.
    distances = compute_all_distances(embeddings, config)

    if benchmark_mode:
        # ── THRESHOLD SWEEP ──────────────────────────────────────────────
        from src.benchmark.evaluate import evaluate_flags, build_pr_curve

        percentile_values = list(range(
            config["sweep_start"],
            config["sweep_stop"],
        ))

        print(f"\n[Benchmark] Sweeping percentile over {len(percentile_values)} "
              f"values: {percentile_values[0]} -> {percentile_values[-1]}")

        threshold_flag_pairs = []
        for pct in percentile_values:
            flagged = flag_above_percentile(distances, pct)
            threshold_flag_pairs.append((pct, flagged))

        # Build PR curve and find the best threshold.
        curve = build_pr_curve(
            threshold_flag_pairs=threshold_flag_pairs,
            corruption_types=["image_corruption"],
            save_path=os.path.join(REPORTS_DIR, "phase3_pr_curve.png"),
            threshold_label="Percentile",
        )

        best_pct = curve["best_threshold"]
        print(f"\n[Benchmark] Best percentile = {best_pct} "
              f"(F1 = {curve['best_f1']:.4f})")

        # Flag at the best threshold.
        flagged = flag_above_percentile(distances, best_pct)

        # Final evaluation.
        evaluate_flags(
            flagged_indices=flagged,
            corruption_types=["image_corruption"],
        )

    else:
        # ── PRODUCTION MODE ──────────────────────────────────────────────
        pct = config["default_percentile"]
        print(f"\n[Production] Using default percentile = {pct}")
        flagged = flag_above_percentile(distances, pct)

    # ── SAVE CSV ─────────────────────────────────────────────────────────
    save_outliers_csv(flagged, distances, image_paths)

    print(f"\n[Phase 3] Complete. {len(flagged)} outliers flagged.")
