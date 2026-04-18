"""
find_duplicates.py
==================
Phase 2: Near-Duplicate Detection (Epsilon Filter)

WHAT THIS FILE DOES
-------------------
Detects near-duplicate image pairs in the dataset by computing pairwise
cosine similarity on the embedding matrix and flagging pairs whose
similarity exceeds a threshold of 1 − ε.

Two computational paths are supported:
    - EXACT PATH (N < 50,000): Full N×N similarity matrix via numpy dot product.
      Memory cost: N² × 4 bytes. At N=10,000 this is ~400MB — fine.
    - APPROXIMATE PATH (N ≥ 50,000): FAISS IndexFlatIP for approximate
      nearest-neighbour search. Avoids materialising the full N×N matrix,
      which would be ~10GB+ at N=50,000.

The epsilon parameter controls sensitivity:
    - Small ε (e.g. 0.01) → strict threshold (sim > 0.99) → few flags, high precision
    - Large ε (e.g. 0.20) → loose threshold (sim > 0.80) → many flags, high recall

In benchmark mode, ε is swept from 0.01 to 0.20 in steps of 0.01,
and the best threshold is selected at maximum F1.

OUTPUTS
-------
    data/03_reports/duplicates.csv
        Columns: index_a, index_b, similarity, path_a, path_b

Run via:
    python -m src.main --mode benchmark --phase 2
Never run this file directly.
"""

import os
import csv
import json
import numpy as np

from src.core_math.distances import cosine_similarity_matrix, find_similar_pairs


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DUPLICATE_CONFIG = {
    # Default epsilon for single-threshold detection.
    # 1 − ε = minimum cosine similarity to flag as near-duplicate.
    "default_epsilon":  0.05,

    # Epsilon sweep range for PR curve construction.
    # np.arange(start, stop, step) — 20 thresholds from 0.01 to 0.20.
    "sweep_start":      0.01,
    "sweep_stop":       0.21,   # exclusive upper bound for np.arange
    "sweep_step":       0.01,

    # FAISS switching threshold. N ≥ this triggers approximate search.
    "faiss_threshold_n": 50_000,

    # For FAISS: number of nearest neighbours to retrieve per query.
    # Must be large enough to find all duplicates but small enough to be fast.
    # 50 is generous — true duplicates should be in the top 5 neighbours.
    "faiss_k":          50,
}

# Path constants matching the pipeline filesystem layout.
EMBEDDINGS_PATH = "data/02_embeddings/embeddings.npy"
INDEX_PATH      = "data/02_embeddings/index.json"
REPORTS_DIR     = "data/03_reports"
OUTPUT_CSV      = os.path.join(REPORTS_DIR, "duplicates.csv")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: LOAD INPUTS
# ─────────────────────────────────────────────────────────────────────────────

def load_inputs():
    """
    Load the embedding matrix and image index from disk.

    Returns:
        embeddings  : np.ndarray of shape (N, 768), dtype float32, L2-normalised.
        image_paths : list of N path strings.

    Raises:
        FileNotFoundError if either file is missing.
    """
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"embeddings.npy not found at '{EMBEDDINGS_PATH}'. "
            f"Run Phase 1 first."
        )
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"index.json not found at '{INDEX_PATH}'. "
            f"Run Phase 1 first."
        )

    embeddings = np.load(EMBEDDINGS_PATH)
    with open(INDEX_PATH, "r") as f:
        image_paths = json.load(f)

    print(f"[Phase 2] Loaded embeddings: {embeddings.shape}")
    print(f"[Phase 2] Loaded index: {len(image_paths)} entries")

    return embeddings, image_paths


# ─────────────────────────────────────────────────────────────────────────────
# EXACT PATH: Full N×N pairwise similarity (N < 50,000)
# ─────────────────────────────────────────────────────────────────────────────

def find_duplicates_exact(embeddings, epsilon):
    """
    Exact near-duplicate detection for small-to-medium datasets.

    Computes the full N×N cosine similarity matrix, then extracts all
    pairs above the threshold (1 − ε).

    Args:
        embeddings : (N, 768) L2-normalised float32 array.
        epsilon    : float, the ε parameter. Threshold = 1 − ε.

    Returns:
        pairs       : list of (index_a, index_b, similarity) tuples.
        flagged_set : set of all unique indices appearing in any pair.
    """
    threshold = 1.0 - epsilon

    print(f"\n[Exact] Computing full N×N similarity matrix ...")
    sim_matrix = cosine_similarity_matrix(embeddings)
    print(f"[Exact] Similarity matrix shape: {sim_matrix.shape}")

    print(f"[Exact] Finding pairs with similarity > {threshold:.4f} (ε={epsilon}) ...")
    pairs = find_similar_pairs(sim_matrix, threshold)

    # Build the set of all flagged indices (both sides of each pair).
    flagged_set = set()
    for idx_a, idx_b, _ in pairs:
        flagged_set.add(idx_a)
        flagged_set.add(idx_b)

    print(f"[Exact] Found {len(pairs)} pairs involving {len(flagged_set)} unique images.")

    return pairs, flagged_set


# ─────────────────────────────────────────────────────────────────────────────
# APPROXIMATE PATH: FAISS ANN search (N ≥ 50,000)
# ─────────────────────────────────────────────────────────────────────────────

def find_duplicates_faiss(embeddings, epsilon, k=50):
    """
    Approximate near-duplicate detection for large datasets using FAISS.

    Builds a FAISS inner-product index (IndexFlatIP) and searches for
    the k nearest neighbours of each embedding. Because embeddings are
    L2-normalised, inner product = cosine similarity.

    IndexFlatIP is exact (not approximate) but avoids materialising
    the full N×N matrix. For true approximation at N > 1M, consider
    IndexIVFFlat or IndexHNSW.

    Args:
        embeddings : (N, 768) L2-normalised float32 array.
        epsilon    : float, the ε parameter. Threshold = 1 − ε.
        k          : number of nearest neighbours to retrieve per query.

    Returns:
        pairs       : list of (index_a, index_b, similarity) tuples.
        flagged_set : set of all unique indices appearing in any pair.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "FAISS is required for datasets with N ≥ 50,000. "
            "Install with: pip install faiss-cpu"
        )

    threshold = 1.0 - epsilon
    N, D = embeddings.shape

    print(f"\n[FAISS] Building IndexFlatIP for {N} vectors of dim {D} ...")
    index = faiss.IndexFlatIP(D)

    # FAISS requires contiguous float32 arrays.
    emb_contiguous = np.ascontiguousarray(embeddings, dtype=np.float32)
    index.add(emb_contiguous)
    print(f"[FAISS] Index built. Total vectors: {index.ntotal}")

    print(f"[FAISS] Searching k={k} nearest neighbours ...")
    # similarities shape: (N, k), indices shape: (N, k)
    similarities, indices = index.search(emb_contiguous, k)

    # Extract pairs above threshold, deduplicating (i, j) and (j, i).
    seen = set()
    pairs = []
    flagged_set = set()

    for i in range(N):
        for j_pos in range(k):
            j = int(indices[i, j_pos])
            sim = float(similarities[i, j_pos])

            # Skip self-matches and below-threshold matches.
            if j == i or sim <= threshold:
                continue

            # Deduplicate: store the pair with smaller index first.
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            pairs.append((pair_key[0], pair_key[1], sim))
            flagged_set.add(i)
            flagged_set.add(j)

    # Sort by similarity descending.
    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"[FAISS] Found {len(pairs)} pairs involving "
          f"{len(flagged_set)} unique images.")

    return pairs, flagged_set


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCHER: Choose exact or FAISS path based on dataset size
# ─────────────────────────────────────────────────────────────────────────────

def find_duplicates(embeddings, epsilon, config=None):
    """
    Dispatch to exact or FAISS path based on dataset size.

    Args:
        embeddings : (N, 768) L2-normalised float32 array.
        epsilon    : float, the ε parameter.
        config     : optional config dict (defaults to DUPLICATE_CONFIG).

    Returns:
        pairs       : list of (index_a, index_b, similarity) tuples.
        flagged_set : set of all flagged indices.
    """
    if config is None:
        config = DUPLICATE_CONFIG

    N = embeddings.shape[0]

    if N >= config["faiss_threshold_n"]:
        print(f"[Phase 2] N={N} >= {config['faiss_threshold_n']} -> using FAISS path.")
        return find_duplicates_faiss(embeddings, epsilon, k=config["faiss_k"])
    else:
        print(f"[Phase 2] N={N} < {config['faiss_threshold_n']} -> using exact path.")
        return find_duplicates_exact(embeddings, epsilon)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS TO CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_duplicates_csv(pairs, image_paths, output_path=OUTPUT_CSV):
    """
    Write the flagged duplicate pairs to a CSV file.

    Columns: index_a, index_b, similarity, path_a, path_b

    Args:
        pairs       : list of (index_a, index_b, similarity) tuples.
        image_paths : list of image path strings for resolving indices.
        output_path : CSV output file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index_a", "index_b", "similarity", "path_a", "path_b"])

        for idx_a, idx_b, sim in pairs:
            path_a = image_paths[idx_a] if idx_a < len(image_paths) else "?"
            path_b = image_paths[idx_b] if idx_b < len(image_paths) else "?"
            writer.writerow([idx_a, idx_b, f"{sim:.6f}", path_a, path_b])

    print(f"\n[Save] duplicates.csv -> '{output_path}' ({len(pairs)} pairs)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_duplicate_detection(benchmark_mode=False):
    """
    Orchestrates Phase 2: Near-Duplicate Detection.

    Steps:
        1. Load embeddings and image index.
        2. In benchmark mode: sweep ε, build PR curve, pick best threshold.
           In production mode: use default ε.
        3. Run detection at the chosen threshold.
        4. Save duplicates.csv.

    Args:
        benchmark_mode : bool. If True, sweep thresholds and score against
                         ground_truth.json["duplicate"].
    """
    config = DUPLICATE_CONFIG.copy()
    embeddings, image_paths = load_inputs()

    if benchmark_mode:
        # ── THRESHOLD SWEEP ──────────────────────────────────────────────
        # Run detection at each ε in the sweep range. Collect flagged sets
        # and pass them to build_pr_curve() for evaluation.
        from src.benchmark.evaluate import evaluate_flags, build_pr_curve

        epsilon_values = np.arange(
            config["sweep_start"],
            config["sweep_stop"],
            config["sweep_step"],
        )
        epsilon_values = [round(float(e), 4) for e in epsilon_values]

        print(f"\n[Benchmark] Sweeping eps over {len(epsilon_values)} values: "
              f"{epsilon_values[0]} -> {epsilon_values[-1]}")

        threshold_flag_pairs = []
        for eps in epsilon_values:
            _, flagged = find_duplicates(embeddings, eps, config)
            threshold_flag_pairs.append((eps, flagged))

        # Build PR curve and find the best threshold.
        curve = build_pr_curve(
            threshold_flag_pairs=threshold_flag_pairs,
            corruption_types=["duplicate"],
            save_path=os.path.join(REPORTS_DIR, "phase2_pr_curve.png"),
            threshold_label="ε (epsilon)",
        )

        best_eps = curve["best_threshold"]
        print(f"\n[Benchmark] Best ε = {best_eps} "
              f"(F1 = {curve['best_f1']:.4f})")

        # Re-run at the best threshold to get the final pairs.
        pairs, flagged = find_duplicates(embeddings, best_eps, config)

        # Final evaluation at best threshold.
        evaluate_flags(
            flagged_indices=flagged,
            corruption_types=["duplicate"],
        )

    else:
        # ── PRODUCTION MODE ──────────────────────────────────────────────
        # Use the default ε without scoring.
        eps = config["default_epsilon"]
        print(f"\n[Production] Using default ε = {eps}")
        pairs, flagged = find_duplicates(embeddings, eps, config)

    # ── SAVE CSV ─────────────────────────────────────────────────────────
    save_duplicates_csv(pairs, image_paths)

    print(f"\n[Phase 2] Complete. {len(pairs)} duplicate pairs flagged "
          f"({len(flagged)} unique images).")
