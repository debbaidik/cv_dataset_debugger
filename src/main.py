"""
main.py
=======
Pipeline Orchestrator for CV Dataset Debugger

WHAT THIS FILE DOES
-------------------
This is the ONLY file that should be called directly. It parses command-line
arguments and calls the correct phase functions in the correct order.

Every other file in src/ is a library — it defines functions but does not
execute anything when imported. main.py is the entry point that wires
everything together.

USAGE
-----
    # Full benchmark pipeline: Phase 0 → Phase 1 → Phases 2–5
    python -m src.main --mode benchmark

    # Skip corruption if ground_truth.json already exists
    python -m src.main --mode benchmark --skip-corruption

    # Skip Phase 0 AND Phase 1 (embeddings already exist)
    python -m src.main --mode benchmark --skip-corruption --skip-extraction

    # Run extraction only (on the corrupted dataset)
    python -m src.main --mode extract

    # Production mode — run on raw data, no ground truth scoring
    python -m src.main --mode production

    # Run only a specific phase (requires embeddings to exist)
    python -m src.main --mode benchmark --phase 2
    python -m src.main --mode benchmark --phase 3
    python -m src.main --mode benchmark --phase 4
    python -m src.main --mode benchmark --phase 5

IMPORTANT
---------
    Always run from the project root directory:
        cd c:\\Personal\\VSFiles\\cv_dataset_debugger
        python -m src.main --mode benchmark     ← correct

    Do NOT run this file directly:
        python src\\main.py                      ← WRONG (ModuleNotFoundError)

    The '-m' flag tells Python to treat 'src' as a package, which requires
    the __init__.py files in every subfolder and the project root as the
    working directory.
"""

import argparse
import sys
import os


# ─────────────────────────────────────────────────────────────────────────────
# PATH CONSTANTS
# These match the filesystem layout in the work plan.
# All paths are relative to the project root (where you run python -m src.main).
# ─────────────────────────────────────────────────────────────────────────────

RAW_DATA_DIR    = "data/01_raw"
EMBEDDINGS_DIR  = "data/02_embeddings"
REPORTS_DIR     = "data/03_reports"
BENCHMARK_DIR   = "data/04_benchmark"
CORRUPTED_DIR   = os.path.join(BENCHMARK_DIR, "corrupted")


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# Defines the CLI interface for the pipeline.
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    """
    Build and return the argument parser.

    Three modes are supported:
        benchmark  — Full pipeline with synthetic ground truth scoring.
                     Phases 0 → 1 → 2 → 3 → 4 → 5, all scored against
                     ground_truth.json.
        extract    — Run Phase 1 only (DINOv2 feature extraction).
                     Useful for re-running extraction without corruption.
        production — Run on raw data without any ground truth.
                     Phases 1 → 2 → 3 → 4 → 5, no scoring.

    Optional flags:
        --skip-corruption  — Skip Phase 0 (assumes ground_truth.json exists).
        --skip-extraction  — Skip Phase 1 (assumes embeddings.npy exists).
        --phase N          — Run only Phase N (2, 3, 4, or 5).
                             Requires embeddings to already exist.
    """
    parser = argparse.ArgumentParser(
        prog="cv_dataset_debugger",
        description=(
            "Statistical pipeline for auditing computer vision datasets. "
            "Embeds images via DINOv2, then runs phases to surface "
            "near-duplicates, outliers, label errors, and systemic bias."
        ),
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["benchmark", "extract", "production"],
        help=(
            "Pipeline mode. "
            "'benchmark' = full pipeline with ground truth scoring. "
            "'extract' = Phase 1 only (DINOv2 extraction). "
            "'production' = run on raw data without scoring."
        ),
    )

    parser.add_argument(
        "--skip-corruption",
        action="store_true",
        default=False,
        help=(
            "Skip Phase 0 (benchmark corruption). Use this if "
            "ground_truth.json and corrupted/ already exist from a previous run."
        ),
    )

    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        default=False,
        help=(
            "Skip Phase 1 (DINOv2 extraction). Use this if "
            "embeddings.npy and index.json already exist from a previous run."
        ),
    )

    parser.add_argument(
        "--phase",
        type=int,
        default=None,
        choices=[2, 3, 4, 5],
        help=(
            "Run only this specific phase. Requires embeddings to exist. "
            "Example: --phase 2 runs only near-duplicate detection."
        ),
    )

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION HELPERS
# Pre-flight checks before any phase runs.
# ─────────────────────────────────────────────────────────────────────────────

def validate_raw_data_exists():
    """
    Check that data/01_raw/ exists and contains at least one subfolder.

    Phase 0 needs raw data to build the corrupted copy.
    Production mode needs raw data to extract embeddings from.

    Raises:
        FileNotFoundError with a clear message if the directory is missing
        or empty.
    """
    if not os.path.exists(RAW_DATA_DIR):
        raise FileNotFoundError(
            f"Raw data directory not found: '{RAW_DATA_DIR}'.\n"
            f"Create the directory and add class subfolders with images:\n"
            f"  {RAW_DATA_DIR}/class_a/image_001.jpg\n"
            f"  {RAW_DATA_DIR}/class_b/image_002.jpg"
        )

    # Check for at least one subfolder (class directory).
    subdirs = [
        d for d in os.listdir(RAW_DATA_DIR)
        if os.path.isdir(os.path.join(RAW_DATA_DIR, d))
    ]
    if len(subdirs) == 0:
        raise FileNotFoundError(
            f"No class subfolders found in '{RAW_DATA_DIR}'.\n"
            f"Expected structure: {RAW_DATA_DIR}/<class_name>/<images>"
        )

    print(f"[Pre-flight] [OK] Raw data exists: {len(subdirs)} class folder(s) in '{RAW_DATA_DIR}'.")


def validate_benchmark_exists():
    """
    Check that Phase 0 outputs exist (ground_truth.json + corrupted/).

    Called when --skip-corruption is used, or before Phases 2–5 in
    benchmark mode. If these files don't exist, the user needs to run
    Phase 0 first.

    Raises:
        FileNotFoundError if ground_truth.json or corrupted/ is missing.
    """
    gt_path = os.path.join(BENCHMARK_DIR, "ground_truth.json")

    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"ground_truth.json not found at '{gt_path}'.\n"
            f"Run Phase 0 first: python -m src.main --mode benchmark"
        )

    if not os.path.exists(CORRUPTED_DIR):
        raise FileNotFoundError(
            f"Corrupted dataset not found at '{CORRUPTED_DIR}'.\n"
            f"Run Phase 0 first: python -m src.main --mode benchmark"
        )

    print(f"[Pre-flight] [OK] Benchmark exists: ground_truth.json + corrupted/")


def validate_embeddings_exist():
    """
    Check that Phase 1 outputs exist (embeddings.npy + index.json).

    Called when --skip-extraction is used, or before Phases 2–5.
    If these files don't exist, the user needs to run Phase 1 first.

    Raises:
        FileNotFoundError if either file is missing.
    """
    emb_path   = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
    index_path = os.path.join(EMBEDDINGS_DIR, "index.json")

    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"embeddings.npy not found at '{emb_path}'.\n"
            f"Run Phase 1 first: python -m src.main --mode extract"
        )

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"index.json not found at '{index_path}'.\n"
            f"Run Phase 1 first: python -m src.main --mode extract"
        )

    print(f"[Pre-flight] [OK] Embeddings exist: embeddings.npy + index.json")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE RUNNERS
# Each function wraps the import + call for one phase.
#
# Imports are deferred (inside the function, not at the top of the file).
# This is intentional:
#   - Phase 0 doesn't need torch at all. Importing torch takes ~3 seconds
#     and fails if CUDA isn't configured. Deferring means Phase 0 can run
#     on a CPU-only machine.
#   - Phase 1 imports torch but not scikit-learn or cleanlab.
#   - Phases 4–5 import cleanlab but not torch.
#   - If a user runs --phase 2, they shouldn't pay the import cost of
#     every other phase's dependencies.
# ─────────────────────────────────────────────────────────────────────────────

def run_phase_0():
    """
    Phase 0: Synthetic Benchmark Construction.

    Imports corrupt_dataset.py and calls build_benchmark().
    This creates data/04_benchmark/corrupted/ and ground_truth.json.

    Requires: data/01_raw/ with class subfolders.
    Produces: data/04_benchmark/ground_truth.json + corrupted/
    """
    print("\n" + "=" * 60)
    print("  PHASE 0: Synthetic Benchmark Construction")
    print("=" * 60)

    from src.benchmark.corrupt_dataset import build_benchmark
    build_benchmark()


def run_phase_1(input_dir, output_dir):
    """
    Phase 1: DINOv2 Feature Extraction.

    Imports dinov2_extractor.py and calls run_extraction().
    Produces embeddings.npy (N×768 L2-normalised matrix) and index.json.

    Args:
        input_dir  : directory containing images to extract from.
                     In benchmark mode → data/04_benchmark/corrupted/
                     In production mode → data/01_raw/
        output_dir : directory to write embeddings.npy + index.json.
                     Always data/02_embeddings/

    Requires: CUDA GPU available.
    Produces: data/02_embeddings/embeddings.npy + index.json
    """
    print("\n" + "=" * 60)
    print("  PHASE 1: DINOv2 Feature Extraction")
    print("=" * 60)
    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_dir}")

    from src.extraction.dinov2_extractor import run_extraction
    run_extraction(input_dir=input_dir, output_dir=output_dir)


def run_phase_2(benchmark_mode=False):
    """
    Phase 2: Near-Duplicate Detection (Epsilon Filter).

    Computes pairwise cosine similarity on the embedding matrix and flags
    pairs that exceed a threshold (1 − ε). In benchmark mode, the results
    are scored against ground_truth.json.

    Requires: data/02_embeddings/embeddings.npy + index.json
    Produces: data/03_reports/duplicates.csv
    """
    print("\n" + "=" * 60)
    print("  PHASE 2: Near-Duplicate Detection")
    print("=" * 60)

    from src.analysis.find_duplicates import run_duplicate_detection
    run_duplicate_detection(benchmark_mode=benchmark_mode)


def run_phase_3(benchmark_mode=False):
    """
    Phase 3: Structural Outlier Detection (Mahalanobis Distance).

    Computes Mahalanobis distance for every embedding from the distribution
    centroid. Images in the statistical tail are flagged as outliers.
    In benchmark mode, scored against ground_truth.json["image_corruption"].

    Requires: data/02_embeddings/embeddings.npy
    Produces: data/03_reports/outliers.csv
    """
    print("\n" + "=" * 60)
    print("  PHASE 3: Structural Outlier Detection")
    print("=" * 60)

    from src.analysis.find_outliers import run_outlier_detection
    run_outlier_detection(benchmark_mode=benchmark_mode)


def run_phase_4(benchmark_mode=False):
    """
    Phase 4: Label Error Detection (Confident Learning).

    Trains a cross-validated classifier on embeddings, produces out-of-fold
    softmax probabilities, and feeds them to cleanlab to find label issues.
    In benchmark mode, scored against ground_truth.json["label_noise"].

    Requires: data/02_embeddings/embeddings.npy + index.json
    Produces: data/03_reports/label_errors.csv
    """
    print("\n" + "=" * 60)
    print("  PHASE 4: Label Error Detection")
    print("=" * 60)

    from src.analysis.find_mislabeled import run_mislabel_detection
    run_mislabel_detection(benchmark_mode=benchmark_mode)


def run_phase_5():
    """
    Phase 5: Bias Discovery (DBSCAN Clustering).

    Clusters the flagged errors from Phases 3 and 4 in embedding space
    using DBSCAN, then compares their clustering density against an
    equal-sized sample of clean (unflagged) embeddings.

    This phase produces a testable claim:
        "Errors cluster more tightly than random → systemic data flaws."
        or
        "Errors are diffuse (ratio ≈ 1.0) → random noise."

    No ground truth scoring — the output IS the finding.

    Requires: data/03_reports/outliers.csv + label_errors.csv + embeddings.npy
    Produces: data/03_reports/error_clusters.csv
    """
    print("\n" + "=" * 60)
    print("  PHASE 5: Bias Discovery (Clustering)")
    print("=" * 60)

    from src.analysis.cluster_errors import run_error_clustering
    run_error_clustering()


# ─────────────────────────────────────────────────────────────────────────────
# MODE HANDLERS
# Each mode defines which phases run and in what order.
# ─────────────────────────────────────────────────────────────────────────────

def handle_benchmark(args):
    """
    Benchmark mode: the full pipeline with ground truth scoring.

    Execution order:
        Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

    The --skip-corruption flag skips Phase 0 (assumes benchmark exists).
    The --skip-extraction flag skips Phase 1 (assumes embeddings exist).
    The --phase N flag runs only that specific phase.

    All analysis phases (2–5) receive benchmark_mode=True, which tells
    them to score their results against ground_truth.json.
    """
    # ── SINGLE PHASE MODE ──────────────────────────────────────────────
    # If --phase is specified, run only that phase and exit.
    # Requires both benchmark and embeddings to already exist.
    if args.phase is not None:
        validate_benchmark_exists()
        validate_embeddings_exist()

        phase_map = {
            2: lambda: run_phase_2(benchmark_mode=True),
            3: lambda: run_phase_3(benchmark_mode=True),
            4: lambda: run_phase_4(benchmark_mode=True),
            5: run_phase_5,
        }
        phase_map[args.phase]()
        return

    # ── PHASE 0: BENCHMARK CONSTRUCTION ────────────────────────────────
    if args.skip_corruption:
        print("\n[Skip] Phase 0 skipped (--skip-corruption). "
              "Using existing benchmark.")
        validate_benchmark_exists()
    else:
        validate_raw_data_exists()
        run_phase_0()

    # ── PHASE 1: FEATURE EXTRACTION ────────────────────────────────────
    # In benchmark mode, extract from the corrupted dataset copy,
    # NOT from data/01_raw/. The corrupted dataset is what we audit.
    if args.skip_extraction:
        print("\n[Skip] Phase 1 skipped (--skip-extraction). "
              "Using existing embeddings.")
        validate_embeddings_exist()
    else:
        run_phase_1(
            input_dir=CORRUPTED_DIR,
            output_dir=EMBEDDINGS_DIR,
        )

    # ── PHASES 2–5: ANALYSIS ──────────────────────────────────────────
    # All phases receive benchmark_mode=True for ground truth scoring.
    run_phase_2(benchmark_mode=True)
    run_phase_3(benchmark_mode=True)
    run_phase_4(benchmark_mode=True)
    run_phase_5()


def handle_extract(args):
    """
    Extract mode: run Phase 1 only.

    Extracts DINOv2 embeddings from the corrupted benchmark dataset.
    If you want to extract from raw data instead, use production mode.

    This mode is useful when you want to re-run extraction without
    re-running Phase 0 (e.g. after changing batch_size or the model).
    """
    validate_benchmark_exists()
    run_phase_1(
        input_dir=CORRUPTED_DIR,
        output_dir=EMBEDDINGS_DIR,
    )


def handle_production(args):
    """
    Production mode: run on raw data without ground truth scoring.

    Execution order:
        Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

    Phase 0 is skipped entirely — there is no synthetic benchmark.
    Phases 2–4 receive benchmark_mode=False, so they flag anomalies
    but do NOT attempt to score against ground_truth.json.
    Phase 5 runs normally (it never uses ground truth).

    Use this mode when auditing a real dataset for the first time.
    """
    validate_raw_data_exists()

    # ── PHASE 1: EXTRACT FROM RAW DATA ─────────────────────────────────
    # In production mode, embeddings come from data/01_raw/,
    # not from any corrupted copy.
    if args.skip_extraction:
        print("\n[Skip] Phase 1 skipped (--skip-extraction). "
              "Using existing embeddings.")
        validate_embeddings_exist()
    else:
        run_phase_1(
            input_dir=RAW_DATA_DIR,
            output_dir=EMBEDDINGS_DIR,
        )

    # ── PHASES 2–5: ANALYSIS (no scoring) ──────────────────────────────
    run_phase_2(benchmark_mode=False)
    run_phase_3(benchmark_mode=False)
    run_phase_4(benchmark_mode=False)
    run_phase_5()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# This block runs when you call: python -m src.main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Parse arguments and dispatch to the correct mode handler.

    The mode handler is a simple dict lookup — each mode maps to a
    function that defines the phase execution order and parameters.
    """
    parser = build_parser()
    args   = parser.parse_args()

    # Print a header with the resolved configuration.
    print("=" * 60)
    print("  CV Dataset Debugger")
    print("=" * 60)
    print(f"  Mode             : {args.mode}")
    print(f"  Skip corruption  : {args.skip_corruption}")
    print(f"  Skip extraction  : {args.skip_extraction}")
    print(f"  Single phase     : {args.phase or 'all'}")
    print(f"  Working directory: {os.getcwd()}")
    print("=" * 60)

    # ── INPUT VALIDATION ──────────────────────────────────────────────
    # Catch contradictory flags before any work starts.

    # --skip-corruption only makes sense in benchmark mode.
    # In production mode, there is no benchmark to skip.
    if args.skip_corruption and args.mode != "benchmark":
        parser.error(
            "--skip-corruption is only valid with --mode benchmark. "
            "Production mode has no Phase 0."
        )

    # --phase only makes sense in benchmark mode.
    # In production mode, all phases run without scoring.
    if args.phase is not None and args.mode == "production":
        parser.error(
            "--phase is only valid with --mode benchmark. "
            "Production mode runs all phases."
        )

    # --phase conflicts with --skip-extraction=False when phases 2-5
    # are selected but embeddings might not exist yet.
    # The validate_embeddings_exist() call inside handle_benchmark
    # will catch this, but it's clearer to warn early.

    # ── DISPATCH ──────────────────────────────────────────────────────
    # Map each mode string to its handler function.
    mode_handlers = {
        "benchmark":  handle_benchmark,
        "extract":    handle_extract,
        "production": handle_production,
    }

    handler = mode_handlers[args.mode]

    try:
        handler(args)
    except FileNotFoundError as e:
        # Missing data files — print a helpful message instead of a traceback.
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except FileExistsError as e:
        # Phase 0 guard: ground_truth.json already exists.
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except RuntimeError as e:
        # GPU not available (from check_gpu in dinov2_extractor.py).
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    # ── DONE ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
