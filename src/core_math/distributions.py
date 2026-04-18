"""
distributions.py
================
Covariance estimation and inversion for Phase 3 (Mahalanobis distance).

This file provides three things:
    1. compute_centroid()      — mean embedding vector (μ)
    2. compute_covariance()    — covariance matrix (Σ) with regularisation
    3. invert_covariance()     — inverse covariance matrix (Σ⁻¹)

Together, these produce the two inputs required by mahalanobis_distances()
in distances.py: the centroid μ and the inverse covariance Σ⁻¹.

WHY COVARIANCE MATTERS
----------------------
The covariance matrix Σ captures how the 768 embedding dimensions co-vary
across all images. It encodes the "shape" of the embedding distribution —
which directions in 768-dimensional space are stretched (high variance)
and which are compressed (low variance).

Mahalanobis distance uses Σ⁻¹ to normalise distances by this shape, so
that "far away" means "statistically unusual" rather than "large number
in some arbitrary dimension".

REGULARISATION
--------------
Σ is a 768×768 matrix. For it to be invertible, we need:
    1. N > 768    — more samples than dimensions (otherwise Σ is rank-deficient)
    2. No perfectly correlated dimensions (otherwise Σ is singular)

If either condition fails, we add a small identity matrix:

    Σ_reg = Σ + λI

where λ = 1e-6 (a tiny value that does not meaningfully change distances
but makes the matrix invertible). This is called Tikhonov regularisation
(or ridge regularisation in the ML world).

The work plan specifies: "If N < 768 or Σ singular: Σ_reg = Σ + 1e-6·I"

WHY ViT-B/14 (768 dimensions)?
-------------------------------
DINOv2 ViT-Small produces 384-dim embeddings. For CIFAR-10 with 10,000
images per class, N ≈ 10,000 > 384 — this would be fine. But for smaller
subsets or per-class analysis, N could drop below 384, making Σ rank-deficient
even with regularisation.

ViT-Base at 768 dimensions is the sweet spot: rich enough for semantic
discrimination, small enough that N > 768 is easy to satisfy with
standard datasets (CIFAR-10: 50,000 training images).

Run via:
    from src.core_math.distributions import compute_centroid, compute_covariance, invert_covariance
Never run this file directly.
"""

import numpy as np
from numpy.linalg import LinAlgError


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT REGULARISATION CONSTANT
# ─────────────────────────────────────────────────────────────────────────────

# λ = 1e-6 is the regularisation strength.
# Small enough to not distort distances meaningfully.
# Large enough to rescue a near-singular covariance matrix.
# If you find that inversion still fails (condition number > 1e12),
# increase this to 1e-5 or 1e-4 — but document why.
DEFAULT_REGULARISATION = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1: COMPUTE CENTROID
# ─────────────────────────────────────────────────────────────────────────────

def compute_centroid(embeddings):
    """
    Compute the centroid (mean vector) of all embeddings.

    The centroid μ is the "center of mass" of the embedding distribution
    in 768-dimensional space. It is the point that minimises the sum of
    squared Euclidean distances to all embeddings.

    Args:
        embeddings : np.ndarray of shape (N, D), dtype float32.
                     N = number of images, D = 768.

    Returns:
        centroid : np.ndarray of shape (D,), dtype float64.
                   The mean embedding vector.

    Note:
        The centroid itself is NOT L2-normalised. The mean of unit vectors
        is generally NOT a unit vector. This is correct — Mahalanobis distance
        measures deviation from the mean, and the mean's norm is irrelevant.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D array (N, D), got shape {embeddings.shape}."
        )

    centroid = np.mean(embeddings, axis=0)

    print(f"[Centroid] Computed from {embeddings.shape[0]} embeddings. "
          f"Centroid norm: {np.linalg.norm(centroid):.6f}")

    return centroid


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2: COMPUTE COVARIANCE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def compute_covariance(embeddings, regularisation=DEFAULT_REGULARISATION):
    """
    Compute the covariance matrix of all embeddings, with regularisation.

    The covariance matrix Σ is computed as:

        Σ = (1 / (N-1)) · (X − μ)ᵀ (X − μ)

    where X is the (N, D) embedding matrix and μ is the centroid.
    np.cov(embeddings.T) computes this with the N-1 denominator (Bessel's
    correction for unbiased estimation).

    Regularisation is always applied:

        Σ_reg = Σ + λI

    where λ is the regularisation constant and I is the D×D identity matrix.
    This ensures invertibility even if:
        - N < D (rank-deficient: not enough samples for the matrix to be full rank)
        - Some dimensions are perfectly correlated (determinant = 0)
        - Floating point accumulation makes the matrix numerically singular

    The regularisation is unconditional (always applied, not just when needed)
    because:
        1. It costs nothing (adding a scalar to the diagonal is O(D)).
        2. It never hurts (λ = 1e-6 changes distances by < 0.001%).
        3. It prevents intermittent failures on edge-case datasets.

    Args:
        embeddings      : np.ndarray of shape (N, D), dtype float32.
        regularisation  : float, the λ value. Default: 1e-6.

    Returns:
        cov_matrix : np.ndarray of shape (D, D), dtype float64.
                     The regularised covariance matrix.

    Shape notes:
        np.cov expects variables as ROWS, observations as columns.
        Our embeddings are (N, D) — N observations of D variables.
        So we pass embeddings.T, which is (D, N), giving a (D, D) output.
    """
    N, D = embeddings.shape

    # Warn if N < D — the covariance matrix is guaranteed rank-deficient.
    # Regularisation will rescue inversion, but the resulting Mahalanobis
    # distances may be less statistically meaningful.
    if N < D:
        print(f"[Covariance] WARNING: N ({N}) < D ({D}). "
              f"Covariance matrix is rank-deficient. "
              f"Regularisation will be applied, but consider using more data.")

    # np.cov computes (D, D) covariance matrix.
    # rowvar=True (default): each row of the input is a variable.
    # We pass embeddings.T so that each of the D dimensions is a row,
    # with N observations per row.
    cov_matrix = np.cov(embeddings.T)

    # Apply regularisation: Σ_reg = Σ + λI
    # np.eye(D) is the D×D identity matrix.
    cov_matrix += regularisation * np.eye(D)

    # Report diagnostics.
    print(f"[Covariance] Shape: {cov_matrix.shape} | "
          f"Regularisation: λ = {regularisation}")
    print(f"[Covariance] Diagonal range: "
          f"[{cov_matrix.diagonal().min():.6f}, {cov_matrix.diagonal().max():.6f}]")

    return cov_matrix


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 3: INVERT COVARIANCE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def invert_covariance(cov_matrix, fallback_regularisation=1e-4):
    """
    Invert the covariance matrix to produce Σ⁻¹.

    Uses np.linalg.inv() which computes the exact matrix inverse.
    The covariance matrix has already been regularised in compute_covariance(),
    so this should succeed on the first attempt.

    If inversion still fails (extremely ill-conditioned matrix), a fallback
    with stronger regularisation is attempted. If that also fails, we raise
    the error — the data likely has a fundamental problem (e.g. all embeddings
    are identical).

    Condition number check:
        After inversion, we compute the condition number κ(Σ).
        κ(Σ) = ||Σ|| · ||Σ⁻¹|| ≈ σ_max / σ_min (ratio of largest to smallest
        singular value).

        κ < 1e6    — well-conditioned (distances are reliable)
        κ ≈ 1e6–12 — mildly ill-conditioned (distances may have precision loss)
        κ > 1e12   — severely ill-conditioned (distances are unreliable)

    Args:
        cov_matrix               : np.ndarray of shape (D, D), regularised covariance.
        fallback_regularisation  : float, stronger λ to try if first inversion fails.

    Returns:
        cov_inv : np.ndarray of shape (D, D), dtype float64.
                  The inverse covariance matrix.

    Raises:
        LinAlgError if both inversion attempts fail.
    """
    D = cov_matrix.shape[0]

    try:
        cov_inv = np.linalg.inv(cov_matrix)

    except LinAlgError:
        # First inversion failed. Apply stronger regularisation and retry.
        print(f"[Invert] WARNING: Inversion failed with initial regularisation. "
              f"Retrying with λ = {fallback_regularisation} ...")

        cov_matrix_strong = cov_matrix + fallback_regularisation * np.eye(D)

        try:
            cov_inv = np.linalg.inv(cov_matrix_strong)
            print(f"[Invert] [OK] Inversion succeeded with fallback regularisation.")

        except LinAlgError:
            raise LinAlgError(
                f"Covariance matrix inversion failed even with "
                f"fallback λ = {fallback_regularisation}. "
                f"The embedding matrix may be degenerate (all rows identical, "
                f"or too few samples for the embedding dimension)."
            )

    # ── CONDITION NUMBER CHECK ──────────────────────────────────────────
    # A high condition number means the matrix is close to singular,
    # and small input perturbations produce large output changes.
    # This makes Mahalanobis distances numerically unstable.
    cond = np.linalg.cond(cov_matrix)

    if cond > 1e12:
        print(f"[Invert] WARNING: Condition number = {cond:.2e} (very high). "
              f"Mahalanobis distances may be unreliable. "
              f"Consider increasing regularisation or reducing embedding dim.")
    elif cond > 1e6:
        print(f"[Invert] Condition number = {cond:.2e} (mildly elevated). "
              f"Distances should be usable but may have some precision loss.")
    else:
        print(f"[Invert] [OK] Condition number = {cond:.2e} (well-conditioned).")

    # Verify the inverse is correct: Σ · Σ⁻¹ ≈ I
    # Check that the product is close to the identity matrix.
    product = cov_matrix @ cov_inv
    identity_error = np.max(np.abs(product - np.eye(D)))
    print(f"[Invert] Verification: max|Σ·Σ⁻¹ − I| = {identity_error:.2e}")

    if identity_error > 1e-3:
        print(f"[Invert] WARNING: Inverse verification error is high. "
              f"Results may be inaccurate.")

    return cov_inv
