"""
cluster_errors.py
=================
Phase 5: Bias Discovery (DBSCAN Clustering)

WHAT THIS FILE DOES
-------------------
Tests the hypothesis that errors flagged by Phases 3 and 4 are NOT randomly
distributed in embedding space — they cluster, indicating systemic data
collection flaws rather than random noise.

The approach:
    1. Load flagged indices from outliers.csv (Phase 3) and label_errors.csv
       (Phase 4). Union them into a single error set.
    2. Extract the corresponding embedding rows → error subset matrix.
    3. Run DBSCAN on the error embeddings to discover clusters.
    4. Sample an equal-sized set of "clean" (unflagged) embeddings as baseline.
    5. Run the same DBSCAN on the clean sample.
    6. Compare mean intra-cluster distances:
       - If errors cluster more tightly → systemic flaws.
       - If ratio ≈ 1.0 → errors are randomly distributed.

WHY DBSCAN
----------
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
discovers cluster count from data density and labels noise points as -1.
This is more appropriate than K-means when we don't know how many clusters
exist and want to test WHETHER errors cluster at all.

OUTPUTS
-------
    data/03_reports/error_clusters.csv
        Columns: index, cluster_id, path, source_phase

    Console output:
        "Errors are clustered (ratio=X) → systemic."
        or
        "Errors are diffuse (ratio=X) → random."

Run via:
    python -m src.main --mode benchmark --phase 5
Never run this file directly.
"""

import os
import csv
import json
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_CONFIG = {
    # DBSCAN parameters.
    # eps: maximum distance between two samples in the same neighbourhood.
    # min_samples: minimum points to form a dense region (core point).
    "dbscan_eps":         0.3,
    "dbscan_min_samples": 5,

    # Distance metric for DBSCAN and intra-cluster distance computation.
    # 'cosine' is natural for L2-normalised embeddings on the unit hypersphere.
    "metric":             "cosine",

    # Random seed for reproducible clean-sample selection.
    "random_seed":        42,
}

# Path constants.
EMBEDDINGS_PATH   = "data/02_embeddings/embeddings.npy"
INDEX_PATH        = "data/02_embeddings/index.json"
OUTLIERS_CSV      = "data/03_reports/outliers.csv"
LABEL_ERRORS_CSV  = "data/03_reports/label_errors.csv"
REPORTS_DIR       = "data/03_reports"
OUTPUT_CSV        = os.path.join(REPORTS_DIR, "error_clusters.csv")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: LOAD INPUTS
# ─────────────────────────────────────────────────────────────────────────────

def load_inputs():
    """
    Load embeddings, image index, and flagged indices from Phases 3 and 4.

    Returns:
        embeddings   : (N, 768) float32 array.
        image_paths  : list of N path strings.
        outlier_idxs : set of indices flagged by Phase 3.
        label_idxs   : set of indices flagged by Phase 4.
    """
    # Load embeddings.
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"embeddings.npy not found at '{EMBEDDINGS_PATH}'. "
            f"Run Phase 1 first."
        )
    embeddings = np.load(EMBEDDINGS_PATH)

    # Load image index.
    image_paths = []
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "r") as f:
            image_paths = json.load(f)

    # Load Phase 3 outlier indices.
    outlier_idxs = set()
    if os.path.exists(OUTLIERS_CSV):
        with open(OUTLIERS_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                outlier_idxs.add(int(row["index"]))
        print(f"[Phase 5] Loaded {len(outlier_idxs)} outlier indices from Phase 3.")
    else:
        print(f"[Phase 5] WARNING: outliers.csv not found. "
              f"Phase 3 may not have been run.")

    # Load Phase 4 label error indices.
    label_idxs = set()
    if os.path.exists(LABEL_ERRORS_CSV):
        with open(LABEL_ERRORS_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_idxs.add(int(row["index"]))
        print(f"[Phase 5] Loaded {len(label_idxs)} label error indices from Phase 4.")
    else:
        print(f"[Phase 5] WARNING: label_errors.csv not found. "
              f"Phase 4 may not have been run.")

    print(f"[Phase 5] Loaded embeddings: {embeddings.shape}")

    return embeddings, image_paths, outlier_idxs, label_idxs


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: COMPUTE MEAN INTRA-CLUSTER DISTANCE
# ─────────────────────────────────────────────────────────────────────────────

def mean_intra_cluster_distance(embeddings_subset, cluster_labels, metric="cosine"):
    """
    Compute the mean pairwise distance WITHIN each cluster, then average
    across all clusters (weighted by cluster size).

    Points labelled -1 (noise) are excluded — they don't belong to any cluster.

    Args:
        embeddings_subset : (M, D) array, the embedding rows for this group.
        cluster_labels    : (M,) integer array from DBSCAN. -1 = noise.
        metric            : distance metric string.

    Returns:
        mean_dist    : float, the weighted mean intra-cluster distance.
        n_clusters   : int, number of clusters found (excluding noise).
        n_noise      : int, number of noise points.
    """
    unique_labels = set(cluster_labels)
    unique_labels.discard(-1)  # Remove noise label.

    n_clusters = len(unique_labels)
    n_noise = int(np.sum(cluster_labels == -1))

    if n_clusters == 0:
        # No clusters found — all points are noise.
        return float("inf"), 0, n_noise

    total_dist = 0.0
    total_points = 0

    for label in unique_labels:
        mask = cluster_labels == label
        cluster_embs = embeddings_subset[mask]
        n_points = cluster_embs.shape[0]

        if n_points < 2:
            continue

        # Compute pairwise distances within this cluster.
        dists = pairwise_distances(cluster_embs, metric=metric)

        # Mean of the upper triangle (exclude diagonal and duplicate pairs).
        upper_mask = np.triu_indices(n_points, k=1)
        mean_d = float(np.mean(dists[upper_mask]))

        total_dist += mean_d * n_points
        total_points += n_points

    if total_points == 0:
        return float("inf"), n_clusters, n_noise

    mean_dist = total_dist / total_points
    return mean_dist, n_clusters, n_noise


# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS TO CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_clusters_csv(error_indices, cluster_labels, image_paths,
                      outlier_idxs, label_idxs, output_path=OUTPUT_CSV):
    """
    Write the clustered error indices to a CSV file.

    Columns: index, cluster_id, path, source_phase

    source_phase indicates which phase(s) flagged this image:
        "phase3"     — flagged only by outlier detection
        "phase4"     — flagged only by label error detection
        "phase3+4"   — flagged by both phases

    Args:
        error_indices  : sorted list of error indices.
        cluster_labels : integer cluster labels from DBSCAN, aligned with error_indices.
        image_paths    : list of image path strings.
        outlier_idxs   : set of Phase 3 flagged indices.
        label_idxs     : set of Phase 4 flagged indices.
        output_path    : CSV output file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "cluster_id", "path", "source_phase"])

        for i, idx in enumerate(error_indices):
            cluster_id = int(cluster_labels[i])
            path = image_paths[idx] if idx < len(image_paths) else "?"

            # Determine source phase.
            in_phase3 = idx in outlier_idxs
            in_phase4 = idx in label_idxs
            if in_phase3 and in_phase4:
                source = "phase3+4"
            elif in_phase3:
                source = "phase3"
            else:
                source = "phase4"

            writer.writerow([idx, cluster_id, path, source])

    print(f"\n[Save] error_clusters.csv -> '{output_path}' "
          f"({len(error_indices)} entries)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_error_clustering():
    """
    Orchestrates Phase 5: Bias Discovery.

    Steps:
        1. Load embeddings and flagged indices from Phases 3 and 4.
        2. Union all flagged indices (deduplicated).
        3. Extract error embeddings and run DBSCAN.
        4. Sample an equal-sized clean set and run the same DBSCAN.
        5. Compare mean intra-cluster distances.
        6. Report whether errors are systemic or random.
        7. Save error_clusters.csv.
    """
    config = CLUSTER_CONFIG.copy()
    np.random.seed(config["random_seed"])

    embeddings, image_paths, outlier_idxs, label_idxs = load_inputs()
    N = embeddings.shape[0]

    # ── STEP 1: UNION OF FLAGGED INDICES ─────────────────────────────────
    all_error_idxs = outlier_idxs | label_idxs
    n_errors = len(all_error_idxs)

    print(f"\n[Phase 5] Error set:")
    print(f"  Phase 3 outliers  : {len(outlier_idxs)}")
    print(f"  Phase 4 label err : {len(label_idxs)}")
    print(f"  Overlap           : {len(outlier_idxs & label_idxs)}")
    print(f"  Union (total)     : {n_errors}")

    if n_errors < config["dbscan_min_samples"]:
        print(f"\n[Phase 5] WARNING: Only {n_errors} error indices found. "
              f"Not enough to cluster (min_samples={config['dbscan_min_samples']}). "
              f"Skipping clustering.")
        return

    # ── STEP 2: EXTRACT ERROR EMBEDDINGS ─────────────────────────────────
    error_indices = sorted(all_error_idxs)
    error_embeddings = embeddings[error_indices]

    print(f"\n[Phase 5] Error embeddings shape: {error_embeddings.shape}")

    # ── STEP 3: DBSCAN ON ERROR SET ──────────────────────────────────────
    print(f"\n[Phase 5] Running DBSCAN on error embeddings "
          f"(eps={config['dbscan_eps']}, min_samples={config['dbscan_min_samples']}, "
          f"metric={config['metric']}) ...")

    dbscan_errors = DBSCAN(
        eps=config["dbscan_eps"],
        min_samples=config["dbscan_min_samples"],
        metric=config["metric"],
    )
    error_cluster_labels = dbscan_errors.fit_predict(error_embeddings)

    error_mean_dist, error_n_clusters, error_n_noise = mean_intra_cluster_distance(
        error_embeddings, error_cluster_labels, metric=config["metric"],
    )

    print(f"\n[Phase 5] Error clustering results:")
    print(f"  Clusters found  : {error_n_clusters}")
    print(f"  Noise points    : {error_n_noise}")
    print(f"  Mean intra-dist : {error_mean_dist:.6f}")

    # ── STEP 4: SAMPLE CLEAN BASELINE ────────────────────────────────────
    # Select the same number of clean (unflagged) embeddings.
    clean_candidates = [i for i in range(N) if i not in all_error_idxs]

    if len(clean_candidates) < n_errors:
        print(f"[Phase 5] WARNING: Not enough clean samples for a fair baseline. "
              f"Available: {len(clean_candidates)}, need: {n_errors}.")
        clean_sample_size = len(clean_candidates)
    else:
        clean_sample_size = n_errors

    clean_indices = sorted(np.random.choice(
        clean_candidates, size=clean_sample_size, replace=False,
    ).tolist())
    clean_embeddings = embeddings[clean_indices]

    print(f"\n[Phase 5] Clean baseline: {clean_sample_size} samples")

    # ── STEP 5: DBSCAN ON CLEAN BASELINE ─────────────────────────────────
    print(f"[Phase 5] Running DBSCAN on clean baseline ...")

    dbscan_clean = DBSCAN(
        eps=config["dbscan_eps"],
        min_samples=config["dbscan_min_samples"],
        metric=config["metric"],
    )
    clean_cluster_labels = dbscan_clean.fit_predict(clean_embeddings)

    clean_mean_dist, clean_n_clusters, clean_n_noise = mean_intra_cluster_distance(
        clean_embeddings, clean_cluster_labels, metric=config["metric"],
    )

    print(f"\n[Phase 5] Clean clustering results:")
    print(f"  Clusters found  : {clean_n_clusters}")
    print(f"  Noise points    : {clean_n_noise}")
    print(f"  Mean intra-dist : {clean_mean_dist:.6f}")

    # ── STEP 6: COMPARE AND REPORT ───────────────────────────────────────
    # Ratio: error_mean / clean_mean
    # < 1.0 → errors are tighter (more clustered) → systemic
    # ≈ 1.0 → errors are similarly spread → random
    # > 1.0 → errors are more spread out → random (unusual)

    if clean_mean_dist > 0 and error_mean_dist != float("inf"):
        ratio = error_mean_dist / clean_mean_dist
    else:
        ratio = float("nan")

    print(f"\n{'=' * 60}")
    print(f"  PHASE 5: BIAS DISCOVERY RESULT")
    print(f"{'=' * 60}")
    print(f"  Error mean intra-cluster dist : {error_mean_dist:.6f}")
    print(f"  Clean mean intra-cluster dist : {clean_mean_dist:.6f}")
    print(f"  Ratio (error / clean)         : {ratio:.4f}" if not np.isnan(ratio)
          else f"  Ratio (error / clean)         : N/A")
    print(f"{'─' * 60}")

    if np.isnan(ratio):
        print(f"  FINDING: Unable to compute ratio. One or both groups produced")
        print(f"           no clusters. Increase data or adjust DBSCAN parameters.")
    elif ratio < 0.85:
        print(f"  FINDING: Errors are CLUSTERED (ratio = {ratio:.4f} < 0.85)")
        print(f"           -> Systemic data collection flaws detected.")
        print(f"           -> Investigate the cluster centers for common patterns.")
    elif ratio > 1.15:
        print(f"  FINDING: Errors are MORE DIFFUSE than clean data "
              f"(ratio = {ratio:.4f} > 1.15)")
        print(f"           -> Errors appear to be randomly distributed.")
    else:
        print(f"  FINDING: Errors are SIMILARLY DISTRIBUTED to clean data "
              f"(ratio = {ratio:.4f} ~= 1.0)")
        print(f"           -> No strong evidence of systemic bias.")

    print(f"{'=' * 60}")

    # ── STEP 7: SAVE CSV ─────────────────────────────────────────────────
    save_clusters_csv(
        error_indices=error_indices,
        cluster_labels=error_cluster_labels,
        image_paths=image_paths,
        outlier_idxs=outlier_idxs,
        label_idxs=label_idxs,
    )

    print(f"\n[Phase 5] Complete. {n_errors} error indices clustered into "
          f"{error_n_clusters} clusters ({error_n_noise} noise points).")
