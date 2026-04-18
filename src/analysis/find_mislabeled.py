"""
find_mislabeled.py
==================
Phase 4: Label Error Detection (Confident Learning)

WHAT THIS FILE DOES
-------------------
Detects mislabeled images in the dataset using Confident Learning via the
cleanlab library. The approach has three steps:

    1. Train a cross-validated linear classifier (LogisticRegression) on the
       DINOv2 embeddings with the given (potentially noisy) labels.
    2. Produce out-of-fold predicted probabilities for every image — these
       are the model's softmax confidence that each image belongs to each class.
    3. Feed the noisy labels and predicted probabilities into cleanlab's
       find_label_issues(), which uses the Confident Learning algorithm to
       identify samples whose given label is likely wrong.

WHY THIS WORKS
--------------
DINOv2 embeddings encode visual semantics. A cat image has an embedding
near other cat embeddings, regardless of what folder it's in. If a cat
image is in the "dog" folder, a linear classifier trained on embeddings
will predict "cat" with high confidence for that image — conflicting with
its given label "dog". cleanlab detects exactly this pattern.

The cross-validation is critical: if we trained on all data and predicted
on the same data, the model would memorise the wrong labels and miss the
errors. Out-of-fold prediction ensures each image is classified by a model
that never saw it during training.

LABEL HANDLING (BENCHMARK MODE)
-------------------------------
In benchmark mode, Phase 0 moved images into wrong folders. Phase 1
extracted embeddings from the corrupted dataset. The paths in index.json
point to the corrupted folder structure where files already sit in their
wrong folders. So extracting the parent folder name from each path gives
us the noisy labels directly — this is exactly what cleanlab needs.

OUTPUTS
-------
    data/03_reports/label_errors.csv
        Columns: index, given_label, predicted_label, confidence, path

Run via:
    python -m src.main --mode benchmark --phase 4
Never run this file directly.
"""

import os
import csv
import json
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MISLABEL_CONFIG = {
    # Cross-validation folds. 5 is the standard choice.
    # More folds = better probability estimates but slower.
    "cv_folds":     5,

    # LogisticRegression parameters.
    # max_iter=1000 is generous — convergence usually happens around 200–400.
    # solver='lbfgs' is the default and works well for multiclass problems.
    # C=1.0 is the default regularisation strength.
    "max_iter":     1000,
    "solver":       "lbfgs",
    "C":            1.0,
}

# Path constants.
EMBEDDINGS_PATH = "data/02_embeddings/embeddings.npy"
INDEX_PATH      = "data/02_embeddings/index.json"
REPORTS_DIR     = "data/03_reports"
OUTPUT_CSV      = os.path.join(REPORTS_DIR, "label_errors.csv")


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
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"index.json not found at '{INDEX_PATH}'. "
            f"Run Phase 1 first."
        )

    embeddings = np.load(EMBEDDINGS_PATH)
    with open(INDEX_PATH, "r") as f:
        image_paths = json.load(f)

    print(f"[Phase 4] Loaded embeddings: {embeddings.shape}")
    print(f"[Phase 4] Loaded index: {len(image_paths)} entries")

    return embeddings, image_paths


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: EXTRACT LABELS FROM PATHS
# ─────────────────────────────────────────────────────────────────────────────

def extract_labels_from_paths(image_paths):
    """
    Extract the class label for each image from its file path.

    The label is the name of the parent directory. For example:
        data/04_benchmark/corrupted/cat/image_001.jpg  →  "cat"
        data/01_raw/dog/image_042.png                  →  "dog"

    In benchmark mode, Phase 0 moved mislabeled images into wrong folders.
    So the folder name IS the noisy label — exactly what cleanlab needs.

    Args:
        image_paths : list of file path strings.

    Returns:
        labels : list of string labels, one per image.
    """
    labels = []
    for path in image_paths:
        # os.path.dirname gives the parent directory path.
        # os.path.basename on that gives just the folder name.
        parent_dir = os.path.dirname(path)
        label = os.path.basename(parent_dir)
        labels.append(label)

    unique_labels = sorted(set(labels))
    print(f"[Phase 4] Extracted {len(unique_labels)} unique labels: {unique_labels}")

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# CORE: CROSS-VALIDATED PREDICTION + CLEANLAB
# ─────────────────────────────────────────────────────────────────────────────

def detect_label_errors(embeddings, string_labels, config=None):
    """
    Run the Confident Learning pipeline to detect label errors.

    Steps:
        1. Encode string labels → integer labels.
        2. Train LogisticRegression with cross-validation to get
           out-of-fold predicted probabilities.
        3. Feed labels + probabilities into cleanlab to detect issues.

    Args:
        embeddings    : (N, 768) float32 array.
        string_labels : list of N string labels (e.g. ["cat", "dog", ...]).
        config        : optional config dict.

    Returns:
        issue_mask    : boolean array of shape (N,). True = suspected label error.
        pred_probs    : (N, C) predicted probability matrix.
        label_encoder : fitted LabelEncoder for decoding integer labels.
        int_labels    : integer-encoded label array.
    """
    if config is None:
        config = MISLABEL_CONFIG

    # Step 1: Encode labels.
    label_encoder = LabelEncoder()
    int_labels = label_encoder.fit_transform(string_labels)
    n_classes = len(label_encoder.classes_)

    print(f"\n[Phase 4] Label encoding:")
    print(f"  Classes : {list(label_encoder.classes_)}")
    print(f"  N       : {len(int_labels)}")
    print(f"  C       : {n_classes}")

    # Step 2: Cross-validated prediction.
    # cross_val_predict with method="predict_proba" produces out-of-fold
    # softmax probabilities. Each image's probability is predicted by a model
    # that was NOT trained on that image — critical for detecting errors.
    print(f"\n[Phase 4] Running {config['cv_folds']}-fold cross-validated "
          f"LogisticRegression ...")

    clf = LogisticRegression(
        max_iter=config["max_iter"],
        solver=config["solver"],
        C=config["C"],
        random_state=42,
        n_jobs=-1,  # Use all available CPU cores.
    )

    pred_probs = cross_val_predict(
        clf,
        embeddings,
        int_labels,
        cv=config["cv_folds"],
        method="predict_proba",
        n_jobs=-1,
    )

    print(f"[Phase 4] Predicted probabilities shape: {pred_probs.shape}")

    # Step 3: cleanlab label issue detection.
    print(f"\n[Phase 4] Running cleanlab find_label_issues ...")

    from cleanlab.filter import find_label_issues

    issue_mask = find_label_issues(
        labels=int_labels,
        pred_probs=pred_probs,
        return_indices_ranked_by=None,  # Return boolean mask, not indices.
    )

    n_issues = int(np.sum(issue_mask))
    print(f"[Phase 4] cleanlab flagged {n_issues} potential label errors "
          f"({100 * n_issues / len(int_labels):.1f}% of dataset)")

    return issue_mask, pred_probs, label_encoder, int_labels


# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS TO CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_label_errors_csv(flagged_indices, string_labels, pred_probs,
                          label_encoder, int_labels, image_paths,
                          output_path=OUTPUT_CSV):
    """
    Write the flagged label errors to a CSV file.

    Columns: index, given_label, predicted_label, confidence, path

    For each flagged image:
        - given_label     : the label from the folder name (potentially wrong)
        - predicted_label : the model's top prediction (likely the correct label)
        - confidence      : the model's confidence in its prediction

    Args:
        flagged_indices : list or array of flagged integer indices.
        string_labels   : list of string labels.
        pred_probs      : (N, C) predicted probability matrix.
        label_encoder   : fitted LabelEncoder.
        int_labels      : integer-encoded label array.
        image_paths     : list of image path strings.
        output_path     : CSV output file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sort flagged indices by confidence in the predicted label (descending).
    # High confidence = model is very sure the given label is wrong.
    def sort_key(idx):
        predicted_class = int(np.argmax(pred_probs[idx]))
        return pred_probs[idx, predicted_class]

    sorted_indices = sorted(flagged_indices, key=sort_key, reverse=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "given_label", "predicted_label",
                         "confidence", "path"])

        for idx in sorted_indices:
            given_label = string_labels[idx]
            predicted_int = int(np.argmax(pred_probs[idx]))
            predicted_label = label_encoder.inverse_transform([predicted_int])[0]
            confidence = float(pred_probs[idx, predicted_int])
            path = image_paths[idx] if idx < len(image_paths) else "?"

            writer.writerow([
                idx, given_label, predicted_label,
                f"{confidence:.6f}", path,
            ])

    print(f"\n[Save] label_errors.csv -> '{output_path}' "
          f"({len(sorted_indices)} label errors)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_mislabel_detection(benchmark_mode=False):
    """
    Orchestrates Phase 4: Label Error Detection.

    Steps:
        1. Load embeddings and image index.
        2. Extract labels from file paths.
        3. Run Confident Learning (cross-val + cleanlab).
        4. In benchmark mode: sweep confidence thresholds, build PR curve,
           pick best threshold.
           In production mode: use cleanlab's default mask.
        5. Save label_errors.csv.

    Threshold sweep (benchmark mode):
        cleanlab returns a boolean mask, but the real signal is the
        self-confidence score: the predicted probability assigned to each
        image's given label. Low self-confidence means the model thinks
        the image doesn't belong to its labeled class.

        We sweep a confidence threshold from 0.05 to 0.95. At each
        threshold, images with self-confidence below it are flagged.
        build_pr_curve() picks the max-F1 threshold.

    Args:
        benchmark_mode : bool. If True, sweep thresholds and score against
                         ground_truth.json["label_noise"].
    """
    embeddings, image_paths = load_inputs()

    # Extract labels from folder structure.
    string_labels = extract_labels_from_paths(image_paths)

    # Run the detection pipeline.
    issue_mask, pred_probs, label_encoder, int_labels = detect_label_errors(
        embeddings, string_labels,
    )

    if benchmark_mode:
        # ── BENCHMARK SCORING WITH THRESHOLD SWEEP ───────────────────────
        from src.benchmark.evaluate import evaluate_flags, build_pr_curve

        # Compute self-confidence: for each image, the predicted probability
        # of its given label. Low values → model disagrees with the label.
        self_confidence = np.array([
            pred_probs[i, int_labels[i]] for i in range(len(int_labels))
        ])

        print(f"\n[Phase 4] Self-confidence statistics:")
        print(f"  Min    : {self_confidence.min():.4f}")
        print(f"  Max    : {self_confidence.max():.4f}")
        print(f"  Mean   : {self_confidence.mean():.4f}")
        print(f"  Median : {np.median(self_confidence):.4f}")

        # Sweep confidence thresholds: flag images whose self-confidence
        # is BELOW the threshold value. Higher threshold → more flags.
        confidence_thresholds = [round(t, 2) for t in np.arange(0.05, 1.00, 0.05)]

        print(f"\n[Benchmark] Sweeping {len(confidence_thresholds)} confidence "
              f"thresholds: {confidence_thresholds[0]} -> {confidence_thresholds[-1]}")

        threshold_flag_pairs = []
        for thresh in confidence_thresholds:
            flagged = set(np.where(self_confidence < thresh)[0].tolist())
            threshold_flag_pairs.append((thresh, flagged))

        # Build PR curve and find the best threshold.
        curve = build_pr_curve(
            threshold_flag_pairs=threshold_flag_pairs,
            corruption_types=["label_noise"],
            save_path=os.path.join(REPORTS_DIR, "phase4_pr_curve.png"),
            threshold_label="Self-confidence threshold",
        )

        best_thresh = curve["best_threshold"]
        print(f"\n[Benchmark] Best confidence threshold = {best_thresh} "
              f"(F1 = {curve['best_f1']:.4f})")

        # Flag at the best threshold.
        flagged_indices = set(np.where(self_confidence < best_thresh)[0].tolist())

        # Final evaluation at best threshold.
        evaluate_flags(
            flagged_indices=flagged_indices,
            corruption_types=["label_noise"],
        )

    else:
        # ── PRODUCTION MODE ──────────────────────────────────────────────
        # Use cleanlab's default mask without scoring.
        flagged_indices = set(np.where(issue_mask)[0].tolist())

    # ── SAVE CSV ─────────────────────────────────────────────────────────
    save_label_errors_csv(
        flagged_indices=flagged_indices,
        string_labels=string_labels,
        pred_probs=pred_probs,
        label_encoder=label_encoder,
        int_labels=int_labels,
        image_paths=image_paths,
    )

    print(f"\n[Phase 4] Complete. {len(flagged_indices)} label errors flagged.")

