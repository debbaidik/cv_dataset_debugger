"""
corrupt_dataset.py
==================
Phase 0: Synthetic Benchmark Construction

This script does ONE job: take a clean dataset and produce a corrupted copy
of it, injecting three types of known errors, then writing a ground_truth.json
that records exactly what was broken and where.

ground_truth.json is the answer key for every downstream phase (2–5).
It must be treated as immutable once written.

Run via:
    python -m src.main --mode benchmark
Never run this file directly.
"""

import os
import json
import random
import shutil
import numpy as np
from PIL import Image, ImageFilter


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# Set your corruption rates and seed here.
# CRITICAL: Once you run Phase 0 and ground_truth.json exists, do NOT change
# the seed or rates. Changing them would make a different corrupted dataset
# but your answer key would be wrong. Document this seed in your README.
# ─────────────────────────────────────────────────────────────────────────────

CORRUPTION_CONFIG = {
    "label_noise_rate":   0.15,   # 15% of images get their label flipped
    "image_corrupt_rate": 0.05,   # 5%  of images get Gaussian noise injected
    "duplicate_rate":     0.03,   # 3%  of images get a blurred copy inserted
    "random_seed":        42,     # Fix this. Same seed = same corrupted dataset.
    "gaussian_noise_std": 40,     # Pixel-level noise intensity (0–255 scale).
                                  # 40 is visually obvious but not catastrophic.
    "blur_radius":        0.5,    # Gaussian blur radius applied to duplicates.
                                  # Small enough that duplicates look near-identical.
}

# ─────────────────────────────────────────────────────────────────────────────
# PATH CONSTANTS
# These match the filesystem layout defined in the work plan.
# ─────────────────────────────────────────────────────────────────────────────

RAW_DATA_DIR        = "data/01_raw"
BENCHMARK_DIR       = "data/04_benchmark"
CORRUPTED_DIR       = os.path.join(BENCHMARK_DIR, "corrupted")
GROUND_TRUTH_PATH   = os.path.join(BENCHMARK_DIR, "ground_truth.json")

# Valid image extensions we will process. Anything else is silently skipped.
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: INDEX THE DATASET
# ─────────────────────────────────────────────────────────────────────────────

def index_dataset(raw_dir):
    """
    Walk raw_dir and build a flat list of (image_path, class_label) tuples.

    The folder name directly under raw_dir IS the class label.
    Structure expected:
        data/01_raw/
            cat/
                image_001.jpg
                image_002.jpg
            dog/
                image_003.jpg

    Returns:
        all_images  : list of (absolute_path_str, label_str)
        class_names : sorted list of unique class labels
    """
    all_images = []

    # os.listdir gives us the class subfolder names.
    # We sort them so the index order is deterministic across runs.
    for class_name in sorted(os.listdir(raw_dir)):
        class_dir = os.path.join(raw_dir, class_name)

        # Skip anything that isn't a directory (e.g. stray .DS_Store files).
        if not os.path.isdir(class_dir):
            continue

        for fname in sorted(os.listdir(class_dir)):
            ext = os.path.splitext(fname)[1].lower()

            # Skip non-image files silently.
            if ext not in VALID_EXTENSIONS:
                continue

            full_path = os.path.join(class_dir, fname)
            all_images.append((full_path, class_name))

    class_names = sorted(set(label for _, label in all_images))

    print(f"[Index] Found {len(all_images)} images across {len(class_names)} classes.")
    print(f"[Index] Classes: {class_names}")

    return all_images, class_names


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: COPY CLEAN DATA INTO BENCHMARK FOLDER
# ─────────────────────────────────────────────────────────────────────────────

def copy_clean_dataset(raw_dir, corrupted_dir):
    """
    Copy the entire raw dataset into the benchmark/corrupted/ folder.
    Preserves folder structure (class subfolder names stay identical).

    We corrupt the COPY. data/01_raw/ is never touched.

    Uses shutil.copytree which recursively copies a directory tree.
    The destination must not already exist — copytree enforces this by default,
    which acts as an extra guard against accidental overwrites.
    """
    print(f"[Copy] Copying clean dataset from '{raw_dir}' -> '{corrupted_dir}' ...")
    shutil.copytree(raw_dir, corrupted_dir)
    print(f"[Copy] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: SAMPLE INDICES SAFELY
# Picks n_samples indices that haven't been used yet.
# ─────────────────────────────────────────────────────────────────────────────

def sample_unused_indices(n_samples, total_n, used_indices):
    """
    Randomly sample n_samples unique integers from [0, total_n)
    that are NOT already in used_indices.

    Args:
        n_samples    : how many indices to pick
        total_n      : size of the full pool
        used_indices : set of already-allocated indices

    Returns:
        A list of n_samples unique integers, none of which appear in used_indices.

    Raises:
        ValueError if there aren't enough unused indices available.
    """
    available = [i for i in range(total_n) if i not in used_indices]

    if len(available) < n_samples:
        raise ValueError(
            f"Not enough unused images. Need {n_samples}, only {len(available)} available."
        )

    return random.sample(available, n_samples)


# ─────────────────────────────────────────────────────────────────────────────
# PASS 1: LABEL NOISE
# Moves images into the wrong class folder inside corrupted/.
# ─────────────────────────────────────────────────────────────────────────────

def inject_label_noise(all_images, class_names, corrupted_dir, used_indices, n_corrupt):
    """
    For n_corrupt images: move them from their correct class folder to a
    randomly chosen different class folder. The image content is unchanged —
    only its folder location (and therefore its label) changes.

    This mimics a human annotator assigning the wrong class.

    Args:
        all_images   : the master index list of (original_path, label)
        class_names  : list of all unique class labels
        corrupted_dir: root of the corrupted copy
        used_indices : set tracking already-corrupted image indices (mutated in place)
        n_corrupt    : exact number of images to flip

    Returns:
        List of manifest entries, one per corrupted image.
    """
    print(f"\n[Pass 1 | Label Noise] Injecting into {n_corrupt} images ...")
    manifest_entries = []

    # Sample indices that haven't been used in a prior corruption pass.
    selected = sample_unused_indices(n_corrupt, len(all_images), used_indices)

    for idx in selected:
        original_path, original_label = all_images[idx]

        # Pick a different label at random.
        # The list comprehension filters out the image's own label so we never
        # "flip" it to the same class it already belongs to.
        other_classes = [c for c in class_names if c != original_label]
        injected_label = random.choice(other_classes)

        # Build the current file path inside the corrupted copy.
        # The file still lives under its original label folder at this point.
        fname = os.path.basename(original_path)
        src_path  = os.path.join(corrupted_dir, original_label, fname)
        dest_path = os.path.join(corrupted_dir, injected_label, fname)

        # Move the file to the wrong class folder.
        os.rename(src_path, dest_path)

        manifest_entries.append({
            "index":          idx,
            "type":           "label_noise",
            "original_label": original_label,
            "injected_label": injected_label,
            "image_path":     dest_path,    # where the file now lives
        })

        # Mark this index as used so later passes skip it.
        used_indices.add(idx)

    print(f"[Pass 1 | Label Noise] Done. {len(manifest_entries)} images flipped.")
    return manifest_entries


# ─────────────────────────────────────────────────────────────────────────────
# PASS 2: IMAGE CORRUPTION (Gaussian Noise)
# Overwrites images with a noisy version. Label (folder) stays the same.
# ─────────────────────────────────────────────────────────────────────────────

def inject_image_corruption(all_images, corrupted_dir, used_indices, n_corrupt, noise_std):
    """
    For n_corrupt images: load the image, add Gaussian noise to every pixel,
    clip to [0, 255], and overwrite the file. The image stays in its correct
    class folder — the label is unchanged. Only the pixel content is damaged.

    This mimics a failed download, a broken sensor, or a truncated file.

    Args:
        all_images   : the master index list of (original_path, label)
        corrupted_dir: root of the corrupted copy
        used_indices : set tracking already-corrupted image indices (mutated in place)
        n_corrupt    : exact number of images to corrupt
        noise_std    : standard deviation of Gaussian noise (pixel intensity scale)

    Returns:
        List of manifest entries, one per corrupted image.
    """
    print(f"\n[Pass 2 | Image Corruption] Injecting Gaussian noise into {n_corrupt} images ...")
    manifest_entries = []

    selected = sample_unused_indices(n_corrupt, len(all_images), used_indices)

    for idx in selected:
        original_path, label = all_images[idx]
        fname = os.path.basename(original_path)
        file_path = os.path.join(corrupted_dir, label, fname)

        # Load image as a numpy array for pixel-level manipulation.
        img = Image.open(file_path).convert("RGB")
        img_array = np.array(img, dtype=np.float32)

        # Generate Gaussian noise with mean=0 and the configured std.
        # np.random.normal returns floats in the same shape as the image array.
        noise = np.random.normal(loc=0, scale=noise_std, size=img_array.shape)

        # Add noise and clip to valid uint8 range [0, 255].
        # Without clipping, pixel values would overflow or go negative.
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        # Overwrite the file in the corrupted copy with the noisy version.
        noisy_img = Image.fromarray(noisy_array)
        noisy_img.save(file_path)

        manifest_entries.append({
            "index":          idx,
            "type":           "image_corruption",
            "original_label": label,
            "injected_label": label,   # label unchanged — only pixels are corrupted
            "image_path":     file_path,
        })

        used_indices.add(idx)

    print(f"[Pass 2 | Image Corruption] Done. {len(manifest_entries)} images corrupted.")
    return manifest_entries


# ─────────────────────────────────────────────────────────────────────────────
# PASS 3: NEAR-DUPLICATES
# Creates a blurred copy of each selected image in the same class folder.
# The original is untouched. The copy is the "duplicate" the pipeline must find.
# ─────────────────────────────────────────────────────────────────────────────

def inject_duplicates(all_images, corrupted_dir, used_indices, n_corrupt, blur_radius):
    """
    For n_corrupt images: load the image, apply a mild Gaussian blur, and save
    it as a NEW file (with '_dup' appended to the filename) in the SAME class folder.
    The original file is left untouched.

    The blur makes the duplicate visually near-identical but pixel-different,
    so hash-based deduplication won't catch it — only embedding similarity will.

    This mimics video frame sampling or re-uploaded images.

    Args:
        all_images   : the master index list of (original_path, label)
        corrupted_dir: root of the corrupted copy
        used_indices : set tracking already-corrupted image indices (mutated in place)
        n_corrupt    : exact number of source images to duplicate
        blur_radius  : Gaussian blur radius for the duplicate

    Returns:
        List of manifest entries, one per duplicated image.
    """
    print(f"\n[Pass 3 | Near-Duplicates] Creating {n_corrupt} near-duplicate images ...")
    manifest_entries = []

    selected = sample_unused_indices(n_corrupt, len(all_images), used_indices)

    for idx in selected:
        original_path, label = all_images[idx]
        fname = os.path.basename(original_path)
        src_path = os.path.join(corrupted_dir, label, fname)

        # Build a new filename for the duplicate.
        # e.g.  cat_0042.jpg  →  cat_0042_dup.jpg
        name, ext = os.path.splitext(fname)
        dup_fname = f"{name}_dup{ext}"
        dup_path  = os.path.join(corrupted_dir, label, dup_fname)

        # Load, blur, and save as a new file.
        # GaussianBlur with a small radius creates a near-identical image
        # that will land very close (but not identical) in embedding space.
        img = Image.open(src_path).convert("RGB")
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        blurred_img.save(dup_path)

        manifest_entries.append({
            "index":          idx,
            "type":           "duplicate",
            "original_label": label,
            "injected_label": label,   # same folder — label unchanged
            "original_path":  src_path,
            "duplicate_path": dup_path,
            "image_path":     dup_path,
        })

        # Mark the SOURCE image index as used.
        # The duplicate itself is a new file not present in all_images,
        # so we only need to guard the original from further corruption.
        used_indices.add(idx)

    print(f"[Pass 3 | Near-Duplicates] Done. {len(manifest_entries)} duplicates created.")
    return manifest_entries


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATE THE MANIFEST
# Count check: each corruption type must have exactly the expected number
# of entries. If anything is off, we catch it here before any phase runs.
# ─────────────────────────────────────────────────────────────────────────────

def validate_manifest(manifest, n_total, config):
    """
    Verify that ground_truth.json contains the expected number of entries
    for each corruption type.

    Also checks that no image index appears in more than one corruption group,
    enforcing the "no image gets two corruption types" rule.

    Args:
        manifest : the dict that was written to ground_truth.json
        n_total  : total number of images in the dataset
        config   : CORRUPTION_CONFIG dict

    Raises:
        AssertionError if any count is wrong or any index overlap is found.
    """
    print("\n[Validate] Running manifest count checks ...")

    expected_label   = int(n_total * config["label_noise_rate"])
    expected_corrupt = int(n_total * config["image_corrupt_rate"])
    expected_dupes   = int(n_total * config["duplicate_rate"])

    actual_label   = len(manifest["label_noise"])
    actual_corrupt = len(manifest["image_corruption"])
    actual_dupes   = len(manifest["duplicate"])

    assert actual_label == expected_label, (
        f"Label noise count mismatch: expected {expected_label}, got {actual_label}"
    )
    assert actual_corrupt == expected_corrupt, (
        f"Image corruption count mismatch: expected {expected_corrupt}, got {actual_corrupt}"
    )
    assert actual_dupes == expected_dupes, (
        f"Duplicate count mismatch: expected {expected_dupes}, got {actual_dupes}"
    )

    # Cross-check: extract the index sets for each group and verify
    # that pairwise intersections are all empty.
    label_indices   = set(e["index"] for e in manifest["label_noise"])
    corrupt_indices = set(e["index"] for e in manifest["image_corruption"])
    dupe_indices    = set(e["index"] for e in manifest["duplicate"])

    overlap_lc = label_indices & corrupt_indices
    overlap_ld = label_indices & dupe_indices
    overlap_cd = corrupt_indices & dupe_indices

    assert not overlap_lc, f"Overlap between label_noise and image_corruption: {overlap_lc}"
    assert not overlap_ld, f"Overlap between label_noise and duplicate: {overlap_ld}"
    assert not overlap_cd, f"Overlap between image_corruption and duplicate: {overlap_cd}"

    print(f"[Validate] [OK] label_noise:      {actual_label}  (expected {expected_label})")
    print(f"[Validate] [OK] image_corruption: {actual_corrupt}  (expected {expected_corrupt})")
    print(f"[Validate] [OK] duplicate:        {actual_dupes}  (expected {expected_dupes})")
    print(f"[Validate] [OK] No index overlaps across corruption types.")
    print(f"[Validate] Manifest is valid.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def build_benchmark():
    """
    Orchestrates all of Phase 0:
        1. Guard against accidental overwrite of an existing benchmark.
        2. Set random seeds for reproducibility.
        3. Index the clean dataset.
        4. Copy clean data to the benchmark folder.
        5. Run the three corruption passes in order.
        6. Write ground_truth.json.
        7. Validate the manifest.

    Called by src/main.py when --mode benchmark is passed.
    """

    # ── GUARD ─────────────────────────────────────────────────────────────────
    # If ground_truth.json already exists, refuse to proceed.
    # This prevents accidental corruption of an existing valid benchmark.
    # To start a fresh experiment, manually delete data/04_benchmark/ first.
    if os.path.exists(GROUND_TRUTH_PATH):
        raise FileExistsError(
            f"ground_truth.json already exists at '{GROUND_TRUTH_PATH}'.\n"
            f"Phase 0 has already been run. To start a new experiment, "
            f"manually delete '{BENCHMARK_DIR}/' and re-run."
        )

    # ── SEED ──────────────────────────────────────────────────────────────────
    # Fix both Python's random module and NumPy's RNG.
    # Every random choice in this script flows through these two generators,
    # so fixing them guarantees identical output on every run with the same seed.
    seed = CORRUPTION_CONFIG["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    print(f"[Setup] Random seed fixed to {seed}.")

    # ── INDEX ─────────────────────────────────────────────────────────────────
    all_images, class_names = index_dataset(RAW_DATA_DIR)
    N = len(all_images)

    if N == 0:
        raise ValueError(
            f"No valid images found in '{RAW_DATA_DIR}'. "
            f"Check the path and folder structure."
        )

    if len(class_names) < 2:
        raise ValueError(
            f"Only {len(class_names)} class found. Label noise requires at least 2 classes "
            f"(we need somewhere to flip images TO)."
        )

    # ── COPY ──────────────────────────────────────────────────────────────────
    # Create the benchmark root directory, then copy raw data into the
    # corrupted/ subfolder. The copy is what gets modified from here on.
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    copy_clean_dataset(RAW_DATA_DIR, CORRUPTED_DIR)

    # ── COMPUTE CORRUPTION COUNTS ─────────────────────────────────────────────
    # int() truncates (equivalent to floor for positive numbers).
    # This matches the count check formula in validate_manifest.
    n_label   = int(N * CORRUPTION_CONFIG["label_noise_rate"])
    n_corrupt = int(N * CORRUPTION_CONFIG["image_corrupt_rate"])
    n_dupes   = int(N * CORRUPTION_CONFIG["duplicate_rate"])

    total_to_corrupt = n_label + n_corrupt + n_dupes
    print(f"\n[Config] N={N} | label_noise={n_label} | image_corruption={n_corrupt} | duplicates={n_dupes}")
    print(f"[Config] Total images to be touched: {total_to_corrupt} ({100*total_to_corrupt/N:.1f}% of dataset)")

    if total_to_corrupt > N:
        raise ValueError(
            f"Corruption budget ({total_to_corrupt}) exceeds dataset size ({N}). "
            f"Lower your corruption rates in CORRUPTION_CONFIG."
        )

    # ── SHARED STATE: used_indices ─────────────────────────────────────────────
    # This set is passed into every pass and mutated in place.
    # It is the single source of truth for "which images have already been touched".
    # Each pass checks before selecting and adds to it after selecting.
    used_indices = set()

    # ── THREE CORRUPTION PASSES ───────────────────────────────────────────────
    # Order: label noise → image corruption → duplicates.
    # This order is specified in the work plan and must not be changed —
    # changing it would produce a different used_indices state and
    # therefore different selections even with the same seed.

    label_entries   = inject_label_noise(
        all_images, class_names, CORRUPTED_DIR,
        used_indices, n_label
    )

    corrupt_entries = inject_image_corruption(
        all_images, CORRUPTED_DIR,
        used_indices, n_corrupt,
        CORRUPTION_CONFIG["gaussian_noise_std"]
    )

    dupe_entries    = inject_duplicates(
        all_images, CORRUPTED_DIR,
        used_indices, n_dupes,
        CORRUPTION_CONFIG["blur_radius"]
    )

    # ── ASSEMBLE MANIFEST ─────────────────────────────────────────────────────
    # Group entries by corruption type.
    # Also store metadata so the manifest is self-documenting.
    manifest = {
        "seed":              seed,
        "total_images":      N,
        "corruption_config": CORRUPTION_CONFIG,
        "label_noise":       label_entries,
        "image_corruption":  corrupt_entries,
        "duplicate":         dupe_entries,
    }

    # ── WRITE MANIFEST ────────────────────────────────────────────────────────
    print(f"\n[Manifest] Writing ground_truth.json to '{GROUND_TRUTH_PATH}' ...")
    with open(GROUND_TRUTH_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Manifest] Written successfully.")

    # ── VALIDATE ──────────────────────────────────────────────────────────────
    # Re-read from disk (not from the in-memory dict) to catch any
    # serialisation issues before downstream phases depend on this file.
    with open(GROUND_TRUTH_PATH, "r") as f:
        saved_manifest = json.load(f)

    validate_manifest(saved_manifest, N, CORRUPTION_CONFIG)

    # ── DONE ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 0 complete.")
    print(f"  Corrupted dataset : {CORRUPTED_DIR}")
    print(f"  Ground truth      : {GROUND_TRUTH_PATH}")
    print("=" * 60)
    print("\nNext step: open notebooks/00_benchmark_design.ipynb and visually")
    print("inspect a sample of each corruption type before proceeding to Phase 1.")
    print("\ndata/04_benchmark/ is now READ-ONLY. Do not re-run this script.")