"""
distances.py
============
Mathematical distance and similarity functions used by Phases 2 and 3.

This file provides three things:
    1. cosine_similarity_matrix()    — N×N pairwise similarity (Phase 2, exact path)
    2. find_similar_pairs()          — extract (i, j, sim) triples above a threshold
    3. mahalanobis_distances()       — per-row Mahalanobis distance from centroid (Phase 3)

All functions operate on L2-normalised embeddings. The normalisation is done
in Phase 1 (dinov2_extractor.py) and is a prerequisite — these functions do
NOT re-normalise their inputs.

WHY COSINE SIMILARITY = DOT PRODUCT HERE
-----------------------------------------
For two L2-normalised vectors A and B on the unit hypersphere:

    cosine_similarity(A, B) = (A · B) / (||A|| · ||B||)
                            = (A · B) / (1.0 · 1.0)
                            = A · B

So the full pairwise similarity matrix is simply:

    S = embeddings @ embeddings.T

This is an N×N matrix where S[i][j] = cosine similarity between image i and j.

WHY MAHALANOBIS INSTEAD OF EUCLIDEAN
-------------------------------------
Euclidean distance treats all 768 dimensions equally. But embedding dimensions
are NOT equally informative — some carry high variance (meaningful spread),
others carry noise. Mahalanobis distance weights each dimension by the inverse
of its variance (via the covariance matrix), so high-variance directions get
less weight and low-variance directions get more.

An image that is 3 standard deviations out along a low-variance dimension
is far more suspicious than one that is 3 SDs out along a high-variance
dimension. Mahalanobis captures this; Euclidean does not.

Run via:
    from src.core_math.distances import cosine_similarity_matrix, mahalanobis_distances
Never run this file directly.
"""

import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1: PAIRWISE COSINE SIMILARITY MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity_matrix(embeddings):
    """
    Compute the full N×N pairwise cosine similarity matrix.

    Because all embeddings are L2-normalised (every row has norm = 1.0),
    cosine similarity between row i and row j reduces to their dot product.
    The full matrix is therefore just a matrix multiplication:

        S = embeddings @ embeddings.T

    Memory cost: N² × 4 bytes (float32).
        N =  1,000 →    4 MB    (trivial)
        N = 10,000 →  400 MB    (fine on 16GB RAM)
        N = 50,000 → 10 GB     (use FAISS instead — see find_duplicates.py)

    This function does NOT check dataset size. The caller (find_duplicates.py)
    is responsible for choosing between this function and FAISS based on N.

    Args:
        embeddings : np.ndarray of shape (N, D), dtype float32, L2-normalised.
                     D = 768 for DINOv2 ViT-B/14.

    Returns:
        similarity : np.ndarray of shape (N, N), dtype float32.
                     similarity[i][j] = cosine similarity between image i and j.
                     Diagonal is 1.0 (each image is identical to itself).
                     Matrix is symmetric: similarity[i][j] == similarity[j][i].
    """
    # Validate input shape — must be 2D.
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D array (N, D), got shape {embeddings.shape}."
        )

    similarity = embeddings @ embeddings.T

    # The result should be symmetric. Verify a sample as a sanity check.
    # Floating point rounding can cause tiny asymmetries (order 1e-7).
    # We don't force symmetry — the asymmetry is negligible for thresholding.

    return similarity


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2: FIND PAIRS ABOVE A SIMILARITY THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────

def find_similar_pairs(similarity_matrix, threshold):
    """
    Extract all (i, j) pairs where similarity exceeds a threshold.

    Given the N×N similarity matrix from cosine_similarity_matrix(), find
    every pair of images whose cosine similarity is above the threshold.

    The threshold is typically 1 − ε, where ε is the epsilon parameter
    from Phase 2. For example, ε = 0.05 → threshold = 0.95.

    Self-pairs (i == i) are excluded — an image is always identical to itself,
    which is not interesting. Symmetric pairs are deduplicated — if (3, 7) is
    returned, (7, 3) is not.

    Args:
        similarity_matrix : np.ndarray of shape (N, N), from cosine_similarity_matrix().
        threshold         : float, minimum similarity to be flagged.
                            Pairs with similarity > threshold are returned.

    Returns:
        pairs : list of (index_a, index_b, similarity) tuples, sorted by
                similarity descending. Each pair appears once (i < j).

    Example:
        pairs = find_similar_pairs(sim_matrix, threshold=0.95)
        # → [(12, 45, 0.982), (3, 88, 0.961), ...]
    """
    N = similarity_matrix.shape[0]

    # np.where returns arrays of (row_indices, col_indices) where the
    # condition is True. We use the upper triangle (i < j) to avoid
    # returning both (i, j) and (j, i), and to exclude the diagonal.
    row_idx, col_idx = np.where(
        np.triu(similarity_matrix > threshold, k=1)
    )
    # k=1 means: start one diagonal above the main diagonal.
    # This excludes the diagonal (k=0) and the lower triangle.

    # Build the list of (index_a, index_b, similarity) tuples.
    pairs = []
    for r, c in zip(row_idx, col_idx):
        pairs.append((int(r), int(c), float(similarity_matrix[r, c])))

    # Sort by similarity descending — most suspicious pairs first.
    pairs.sort(key=lambda x: x[2], reverse=True)

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 3: MAHALANOBIS DISTANCES
# ─────────────────────────────────────────────────────────────────────────────

def mahalanobis_distances(embeddings, centroid, cov_inv):
    """
    Compute the Mahalanobis distance of every embedding from the centroid.

    The Mahalanobis distance for a single vector v is:

        D_M(v) = sqrt( (v − μ)ᵀ  Σ⁻¹  (v − μ) )

    where:
        μ   = centroid (mean of all embeddings)
        Σ⁻¹ = inverse of the covariance matrix

    This function vectorises the computation for all N embeddings at once:

        1. diff = embeddings − μ           → (N, 768) matrix of deviations
        2. left = diff @ Σ⁻¹              → (N, 768) — each row is weighted
        3. sq_dist = row-wise dot product of left and diff
                   = sum(left * diff, axis=1)  → (N,) array
        4. distances = sqrt(sq_dist)       → (N,) Mahalanobis distances

    The vectorised form avoids a Python loop over N rows, which would be
    extremely slow for N > 1,000.

    Args:
        embeddings : np.ndarray of shape (N, D), dtype float32.
        centroid   : np.ndarray of shape (D,), the mean embedding vector.
                     Computed by distributions.compute_centroid().
        cov_inv    : np.ndarray of shape (D, D), the inverse covariance matrix.
                     Computed by distributions.invert_covariance().

    Returns:
        distances : np.ndarray of shape (N,), dtype float64.
                    distances[i] = Mahalanobis distance of embedding i from centroid.
                    Higher values = further from the distribution center = more outlier-like.

    Raises:
        ValueError if shapes are incompatible.
    """
    N, D = embeddings.shape

    # Validate shapes.
    if centroid.shape != (D,):
        raise ValueError(
            f"Centroid shape {centroid.shape} does not match embedding dim {D}. "
            f"Expected shape ({D},)."
        )
    if cov_inv.shape != (D, D):
        raise ValueError(
            f"Inverse covariance shape {cov_inv.shape} does not match "
            f"embedding dim {D}. Expected shape ({D}, {D})."
        )

    # Step 1: Compute deviation vectors.
    # diff[i] = embeddings[i] - centroid
    # Broadcasting: (N, D) - (D,) → (N, D)
    diff = embeddings - centroid

    # Step 2: Multiply deviations by the inverse covariance.
    # left[i] = diff[i] @ Σ⁻¹  (matrix multiply each row by Σ⁻¹)
    # Shape: (N, D) @ (D, D) → (N, D)
    left = diff @ cov_inv

    # Step 3: Row-wise dot product of left and diff.
    # This computes (v − μ)ᵀ Σ⁻¹ (v − μ) for each row.
    # np.sum(left * diff, axis=1) is equivalent to np.diag(left @ diff.T)
    # but without constructing the N×N outer product.
    # Shape: (N,)
    squared_distances = np.sum(left * diff, axis=1)

    # Guard against tiny negative values from floating point rounding.
    # The true squared Mahalanobis distance is always ≥ 0.
    # Clipping prevents sqrt of negative numbers producing NaN.
    squared_distances = np.clip(squared_distances, 0, None)

    # Step 4: Take the square root.
    distances = np.sqrt(squared_distances)

    return distances


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: FLAG INDICES ABOVE A PERCENTILE THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────

def flag_above_percentile(distances, percentile):
    """
    Return the set of indices where the distance exceeds the given percentile.

    Used by Phase 3 to flag structural outliers: images whose Mahalanobis
    distance is in the extreme tail of the distribution.

    For example, percentile=95 means: flag the top 5% of images by distance.

    Args:
        distances  : np.ndarray of shape (N,), Mahalanobis distances.
        percentile : float in [0, 100]. Images above this cutoff are flagged.

    Returns:
        flagged : set of integer indices where distances[i] > cutoff.

    Example:
        flagged = flag_above_percentile(mahal_distances, percentile=95)
        # → {12, 45, 88, 102, ...}  (the top 5%)
    """
    cutoff = np.percentile(distances, percentile)
    flagged = set(np.where(distances > cutoff)[0].tolist())

    print(f"  Percentile {percentile} -> cutoff = {cutoff:.4f} -> "
          f"{len(flagged)} flagged ({100*len(flagged)/len(distances):.1f}%)")

    return flagged
