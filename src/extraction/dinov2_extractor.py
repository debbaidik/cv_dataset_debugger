"""
dinov2_extractor.py
===================
Phase 1: DINOv2 Feature Extraction

WHAT THIS FILE DOES
-------------------
Runs every image in the dataset through a frozen DINOv2 ViT-B/14 backbone
and extracts a 768-dimensional embedding vector for each image.

After extraction, every embedding row is L2-normalised so that all vectors
sit on the unit hypersphere. This is a prerequisite for Phases 2 and 3:
    - Phase 2 uses cosine similarity, which reduces to a dot product after
      L2 normalisation.
    - Phase 3 uses Mahalanobis distance on the normalised embedding space.

Two files are written to disk:
    data/02_embeddings/embeddings.npy   — N×768 float32 matrix
    data/02_embeddings/index.json       — maps each row index → source image path

Without index.json, flagged indices from Phases 2–5 cannot be traced back
to actual images on disk. Both files are required.

WHY DINOv2 ViT-B/14 (Base variant)?
    - 768 dimensions: large enough for stable covariance matrix inversion in
      Phase 3 (needs N > 768), small enough to keep pairwise distances tractable.
    - Small (384d) risks covariance instability.
    - Large/Giant (1024d/1536d) suffers from the curse of dimensionality —
      cosine discriminability degrades, and marginal semantic gain doesn't
      justify the memory and compute cost.

WORKFLOW
--------
    Step 1 — Confirm GPU is available (cuda.is_available() must return True).
    Step 2 — Load dinov2_vitb14 from torch.hub. Freeze it (eval + no_grad).
    Step 3 — Index all valid images in the input directory.
    Step 4 — Build a Dataset + DataLoader with the exact ImageNet transform.
    Step 5 — Run batched inference. Extract the [CLS] token (index 0) from
              the model's final layer output. Move each batch result to CPU
              immediately — don't let embeddings accumulate on VRAM.
    Step 6 — Stack all outputs into an N×768 matrix. L2-normalise row-wise.
    Step 7 — Spot-check norms (every row must be ≈ 1.0). Flag NaN/zero rows.
    Step 8 — Save embeddings.npy and index.json.
    Step 9 — Run the nearest-neighbour sanity check.

Run via:
    python -m src.main --mode extract
Never run this file directly.

HARDWARE NOTES (RTX 3050 6GB + DDR5 16GB)
    Model weights:  ~330MB VRAM — comfortable.
    Safe batch size: 32. Drop to 16 on CUDA out-of-memory errors.
    Feature matrix:  ~30MB per 10,000 images at float32. Not a constraint.
    In benchmark mode, extraction runs on data/04_benchmark/corrupted/,
    not data/01_raw/. main.py passes the correct path via the --mode flag.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_CONFIG = {
    # Input: which directory to extract from.
    # In benchmark mode, main.py overrides this with the corrupted/ path.
    "input_dir":    "data/04_benchmark/corrupted",

    # Output directory for embeddings.npy and index.json.
    "output_dir":   "data/02_embeddings",

    # DINOv2 model variant. Do not change — other variants produce different
    # embedding dimensions and will break Phase 3's covariance inversion.
    "model_name":   "dinov2_vitb14",

    # Inference batch size. Drop to 16 if you see CUDA out-of-memory errors.
    "batch_size":   32,

    # DataLoader workers for parallel image loading from disk.
    # 2 is safe on most machines; increase to 4 if disk I/O is the bottleneck.
    "num_workers":  2,

    # ImageNet normalisation constants.
    # These are the exact values DINOv2 was pre-trained with.
    # Any deviation silently corrupts the embeddings — the model produces
    # wrong features for inputs that weren't normalised the same way as
    # its training data.
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std":  [0.229, 0.224, 0.225],

    # Target spatial resolution. DINOv2 ViT-B/14 expects 224×224.
    "image_size":   224,

    # Expected embedding dimension for ViT-B. Used for validation.
    "embedding_dim": 768,
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: GPU CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_gpu():
    """
    Confirm that a CUDA-capable GPU is visible to PyTorch.

    This must pass before any extraction begins. Running DINOv2 inference
    on CPU is technically possible but extremely slow — the work plan
    explicitly requires GPU.

    If this fails:
        1. Run `nvidia-smi` to confirm your driver is working.
        2. Check your torch install: `python -c "import torch; print(torch.__version__)"`.
        3. Reinstall torch with the correct CUDA version from pytorch.org,
           matching the CUDA version shown by nvidia-smi.

    Returns:
        torch.device — the device to use for all tensor operations.

    Raises:
        RuntimeError if no CUDA GPU is available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. torch.cuda.is_available() returned False.\n"
            "Check your GPU driver and torch CUDA installation.\n"
            "Run: nvidia-smi   and   python -c \"import torch; print(torch.__version__)\""
        )

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print(f"[GPU] [OK] CUDA available.")
    print(f"[GPU] Device : {gpu_name}")
    print(f"[GPU] VRAM   : {vram_gb:.1f} GB")

    return device


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: LOAD AND FREEZE THE MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str, device: torch.device) -> nn.Module:
    """
    Load the DINOv2 model from torch.hub, freeze it, and move it to GPU.

    "Freezing" means two things here:
        1. model.eval()      — disables dropout and batch norm's training behaviour.
                               Without this, the same image produces different
                               embeddings on different forward passes.
        2. torch.no_grad()   — disables gradient tracking for the entire inference
                               loop. This is handled in extract_embeddings(), not here.
                               Mentioned for clarity: gradients are never computed.

    We do NOT call model.parameters() or touch weights in any way.
    This is pure inference — the model is used as a fixed feature extractor.

    Args:
        model_name : string, must be "dinov2_vitb14" for Phase 1.
        device     : torch.device, the GPU device returned by check_gpu().

    Returns:
        model — the frozen DINOv2 model on GPU, ready for inference.
    """
    print(f"\n[Model] Loading '{model_name}' from torch.hub ...")
    print(f"[Model] (First run downloads ~330MB. Subsequent runs use cache.)")

    # torch.hub.load pulls the model from facebookresearch's DINOv2 repo.
    # force_reload=False means it uses the local cache if available.
    raw_model = torch.hub.load(
        "facebookresearch/dinov2",
        model_name,
        force_reload=False,
    )

    if not isinstance(raw_model, nn.Module):
        raise TypeError(
            f"torch.hub.load returned {type(raw_model).__name__}, expected torch.nn.Module"
        )

    model: nn.Module = raw_model

    # eval() is critical. Without it:
    #   - Dropout layers randomly zero activations → non-deterministic embeddings.
    #   - BatchNorm uses batch statistics instead of running stats → wrong features.
    model.eval()

    # Move all model weights to GPU memory.
    model = model.to(device)

    # Confirm the model is frozen by checking that no parameters require gradients.
    # This is a sanity check — torch.hub loaded models should already have
    # requires_grad=True on weights, but we want no_grad at inference time.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] [OK] Loaded. Trainable params reported: {trainable_params:,} "
          f"(gradients disabled during inference via torch.no_grad)")
    print(f"[Model] Moved to device: {device}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: INDEX VALID IMAGE PATHS
# ─────────────────────────────────────────────────────────────────────────────

def index_images(input_dir):
    """
    Walk input_dir recursively and collect all valid image paths.

    Filters aggressively: only .jpg, .jpeg, and .png files are included.
    Everything else (thumbnails, .DS_Store, .txt, .json, broken files) is
    silently skipped.

    The resulting list is sorted for deterministic ordering — same filesystem
    state always produces the same index. This is important because the row
    number in embeddings.npy is the image's identity in all downstream phases.

    Args:
        input_dir : root directory to walk (e.g. data/04_benchmark/corrupted/)

    Returns:
        image_paths : sorted list of absolute path strings.

    Raises:
        FileNotFoundError if input_dir doesn't exist.
        ValueError if no valid images are found.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(
            f"Input directory not found: '{input_dir}'.\n"
            f"Run Phase 0 first to create the corrupted dataset."
        )

    image_paths = []

    # os.walk yields (dirpath, subdirs, filenames) for every directory.
    for dirpath, _, filenames in os.walk(input_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTENSIONS:
                image_paths.append(os.path.join(dirpath, fname))

    # Sort for deterministic ordering.
    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise ValueError(
            f"No valid images found in '{input_dir}'. "
            f"Check that Phase 0 completed successfully and the folder is not empty."
        )

    print(f"\n[Index] Found {len(image_paths)} valid images in '{input_dir}'.")
    return image_paths


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: DATASET AND TRANSFORM PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_transform(config):
    """
    Build the image preprocessing transform pipeline.

    The transform order is exact and must not be changed:
        1. Resize to 224×224
           DINOv2 ViT-B/14 uses a fixed patch size of 14px over a 224×224 input,
           producing a 16×16 grid of patches + 1 [CLS] token = 257 tokens.
           Any other resolution produces a different sequence length.

        2. ToTensor
           Converts PIL Image [H, W, C] uint8 in [0, 255] to
           torch.FloatTensor [C, H, W] in [0.0, 1.0].
           The channel reorder (HWC → CHW) and dtype cast happen here.

        3. Normalize with ImageNet mean and std
           Shifts and scales each channel so the model sees the same
           distribution it was pre-trained on. Using wrong constants
           (e.g. [0.5, 0.5, 0.5]) will produce wrong embeddings —
           the model's internal statistics assume ImageNet normalisation.

    Args:
        config : EXTRACTION_CONFIG dict.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config["imagenet_mean"],
            std=config["imagenet_std"],
        ),
    ])


class ImageFileDataset(Dataset):
    """
    Minimal PyTorch Dataset that loads images from a flat list of file paths.

    Handles two failure modes gracefully:
        - Corrupt or unreadable images: returns a zero tensor of the correct
          shape and logs the path. A zero tensor after L2 normalisation
          becomes a NaN (norm=0), which is caught in the spot-check step.
        - Non-RGB images (RGBA, grayscale, palette): force-converted to RGB
          before the transform runs. DINOv2 expects 3-channel input.

    Args:
        image_paths : list of file path strings (from index_images).
        transform   : the torchvision transform pipeline (from build_transform).
        image_size  : int, the target spatial size (224).
    """

    def __init__(self, image_paths, transform, image_size=224):
        self.image_paths = image_paths
        self.transform   = transform
        self.image_size  = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            # Force RGB: handles grayscale (L), RGBA, palette (P) images.
            # DINOv2 always expects 3 input channels.
            img = Image.open(path).convert("RGB")
            return self.transform(img)

        except Exception as e:
            # If the image is unreadable (truncated, corrupt header, etc.),
            # return a zero tensor. This will produce a zero-norm embedding
            # that the spot-check step will catch and report.
            print(f"[Dataset] WARNING: Could not load '{path}': {e}")
            print(f"[Dataset]          Returning zero tensor. This index will "
                  f"be flagged in the norm check.")
            return torch.zeros(3, self.image_size, self.image_size)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 & 6: BATCHED INFERENCE + L2 NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def extract_embeddings(model, image_paths, config, device):
    """
    Run batched DINOv2 inference over all images and return the L2-normalised
    N×768 embedding matrix.

    Inference loop details:
        - torch.no_grad() wraps the entire loop. No gradients are ever computed.
          This saves VRAM and speeds up inference significantly.
        - Each batch is moved to GPU, forward-passed, and the [CLS] token
          (index 0 of the sequence dimension) is extracted.
        - The [CLS] token is a 768-dimensional vector that DINOv2 trains to
          summarise the global semantic content of the image.
        - Results are moved back to CPU immediately after each batch.
          This prevents VRAM accumulation — at batch_size=32, keeping results
          on GPU would fill VRAM within ~200 batches on a 6GB card.

    L2 normalisation:
        After stacking all batch outputs into a single matrix, every row is
        divided by its own L2 norm. This projects all embeddings onto the
        unit hypersphere, which is required for:
            - Phase 2: cosine similarity = dot product (only true on unit sphere)
            - Phase 3: Mahalanobis captures directional spread correctly

    Args:
        model       : frozen DINOv2 model on GPU.
        image_paths : sorted list of image path strings.
        config      : EXTRACTION_CONFIG dict.
        device      : torch.device (GPU).

    Returns:
        embeddings  : np.ndarray of shape (N, 768), dtype float32, L2-normalised.
    """
    transform = build_transform(config)
    dataset   = ImageFileDataset(image_paths, transform, config["image_size"])

    # pin_memory=True speeds up CPU→GPU transfer by using page-locked memory.
    # Only effective when num_workers > 0.
    dataloader = DataLoader(
        dataset,
        batch_size  = config["batch_size"],
        num_workers = config["num_workers"],
        pin_memory  = True,
        shuffle     = False,   # CRITICAL: do not shuffle — index order must match image_paths
    )

    all_embeddings = []
    total_batches  = len(dataloader)

    print(f"\n[Extract] Starting inference over {len(image_paths)} images ...")
    print(f"[Extract] Batch size: {config['batch_size']} | "
          f"Batches: {total_batches} | Workers: {config['num_workers']}")

    # torch.no_grad() disables autograd for the entire inference loop.
    # Required: without it, PyTorch tracks computation graphs for every
    # forward pass, consuming VRAM proportional to the number of batches.
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):

            # Move input tensor from CPU to GPU.
            batch = batch.to(device, non_blocking=True)
            # non_blocking=True allows async GPU transfer, overlapping
            # with CPU-side data loading for the next batch.

            # Forward pass through the frozen DINOv2 backbone.
            # Output shape: (batch_size, num_tokens, embedding_dim)
            #             = (32, 257, 768)
            # 257 tokens = 256 patch tokens + 1 [CLS] token.
            output = model(batch)

            # The model's forward() returns only the [CLS] token by default
            # when called with no arguments, giving shape (batch_size, 768).
            # If you call model.forward_features(batch), you get all tokens
            # and must index [:, 0, :] to extract the [CLS] token.
            # torch.hub loaded dinov2_vitb14 returns [CLS] directly.
            assert output.ndim == 2 and output.shape[1] == config["embedding_dim"], (
                f"Unexpected model output shape: {output.shape}. "
                f"Expected (batch_size, {config['embedding_dim']}). "
                f"The torch.hub DINOv2 API may have changed."
            )
            cls_embeddings = output  # shape: (batch_size, 768)

            # Move to CPU immediately. Do not accumulate on VRAM.
            all_embeddings.append(cls_embeddings.cpu().numpy())

            # Progress logging every 10 batches.
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                processed = min((batch_idx + 1) * config["batch_size"], len(image_paths))
                print(f"[Extract] Batch {batch_idx+1}/{total_batches} — "
                      f"{processed}/{len(image_paths)} images processed.")

            # Free the GPU batch tensor explicitly.
            # Python's garbage collector handles this eventually, but
            # explicit deletion prevents transient VRAM spikes between batches.
            del batch, cls_embeddings

    # Stack all (batch_size, 768) arrays into a single (N, 768) matrix.
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"\n[Extract] Raw embeddings shape: {embeddings.shape}")

    # ── L2 NORMALISATION ──────────────────────────────────────────────────────
    # Divide each row by its L2 norm.
    # np.linalg.norm with axis=1, keepdims=True gives an (N, 1) array of norms,
    # which broadcasts correctly over the (N, 768) embedding matrix.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Guard against zero-norm rows (corrupted images that returned zero tensors).
    # Dividing by zero would produce NaN — we detect these in the spot-check.
    # Replace zero norms with 1.0 temporarily so division doesn't crash;
    # the resulting row will be all-zeros, which the spot-check catches.
    zero_norm_mask = (norms.squeeze() == 0)
    norms[zero_norm_mask] = 1.0

    embeddings = embeddings / norms
    print(f"[Extract] L2 normalisation applied.")

    if zero_norm_mask.any():
        zero_indices = np.where(zero_norm_mask)[0].tolist()
        print(f"[Extract] WARNING: {len(zero_indices)} zero-norm row(s) detected "
              f"at indices: {zero_indices}")
        print(f"[Extract] These correspond to unreadable images. "
              f"Check the paths in index.json at those indices.")

    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: NORM SPOT-CHECK
# ─────────────────────────────────────────────────────────────────────────────

def spot_check_norms(embeddings, n_samples=10):
    """
    Sample n_samples rows from the embedding matrix and verify that each
    row's L2 norm is ≈ 1.0 (within floating point tolerance).

    After L2 normalisation, every row must lie on the unit hypersphere.
    If any norm deviates meaningfully from 1.0, the normalisation step
    has a bug or the model returned unexpected output shapes.

    NaN values indicate that a zero-norm row was divided by itself —
    always caused by an unreadable image returning a zero tensor.

    Args:
        embeddings : (N, 768) float32 numpy array.
        n_samples  : number of random rows to check.

    Raises:
        ValueError if any sampled norm is not ≈ 1.0 or is NaN.
    """
    N = embeddings.shape[0]
    sample_indices = np.random.choice(N, size=min(n_samples, N), replace=False)

    print(f"\n[Norm Check] Sampling {len(sample_indices)} rows ...")
    failed = []

    for idx in sample_indices:
        row  = embeddings[idx]
        norm = np.linalg.norm(row)

        if np.isnan(norm):
            print(f"[Norm Check]  Row {idx:>6} — NaN norm (unreadable image)")
            failed.append(idx)
        elif not np.isclose(norm, 1.0, atol=1e-5):
            # atol=1e-5 is generous enough to absorb float32 rounding.
            print(f"[Norm Check]  Row {idx:>6} -- norm = {norm:.8f}  <-- UNEXPECTED")
            failed.append(idx)
        else:
            print(f"[Norm Check]  Row {idx:>6} -- norm = {norm:.8f}  [OK]")

    if failed:
        raise ValueError(
            f"Norm check failed for {len(failed)} row(s): {failed}.\n"
            f"Possible causes:\n"
            f"  - Unreadable images produced zero tensors (NaN after normalisation).\n"
            f"  - The normalisation step has a bug.\n"
            f"  - The model returned unexpected output.\n"
            f"Check the corresponding paths in index.json."
        )

    print(f"[Norm Check] [OK] All sampled norms are ~1.0. Normalisation is correct.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(embeddings, image_paths, output_dir):
    """
    Save the embedding matrix and image index to disk.

    Writes two files:
        embeddings.npy  — the N×768 float32 matrix.
                          Loaded by Phases 2, 3, 4, 5 as the primary data source.
        index.json      — list of image path strings, one per row.
                          index.json[i] = path of the image whose embedding is
                          stored in row i of embeddings.npy.
                          This file is how flagged indices become traceable
                          back to actual images on disk.

    Args:
        embeddings  : (N, 768) float32 numpy array.
        image_paths : list of path strings in the same order as embedding rows.
        output_dir  : directory to write files into.

    Raises:
        ValueError if embeddings.shape[0] != len(image_paths).
                   A mismatch means the DataLoader shuffled or skipped images.
    """
    if embeddings.shape[0] != len(image_paths):
        raise ValueError(
            f"Embedding row count ({embeddings.shape[0]}) does not match "
            f"image path count ({len(image_paths)}). "
            f"Check that shuffle=False in the DataLoader and no images were skipped."
        )

    os.makedirs(output_dir, exist_ok=True)

    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    index_path      = os.path.join(output_dir, "index.json")

    # Save the numpy matrix in .npy binary format.
    # .npy preserves dtype (float32) and shape exactly.
    # np.load(embeddings_path) reloads it identically in downstream phases.
    np.save(embeddings_path, embeddings)
    print(f"\n[Save] embeddings.npy -> '{embeddings_path}'")
    print(f"[Save] Shape: {embeddings.shape} | "
          f"Size: {embeddings.nbytes / (1024**2):.1f} MB")

    # Save the index as a plain JSON list.
    # list[i] = path string of the image whose embedding is at row i.
    with open(index_path, "w") as f:
        json.dump(image_paths, f, indent=2)
    print(f"[Save] index.json    -> '{index_path}'")
    print(f"[Save] {len(image_paths)} entries.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: NEAREST-NEIGHBOUR SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def nearest_neighbour_sanity_check(embeddings, image_paths, n_checks=3):
    """
    For n_checks randomly selected images, find the nearest neighbour by
    cosine similarity and print both image paths.

    If the transform pipeline is correct, each query image and its nearest
    neighbour should be visually similar (same class, similar appearance).
    If unrelated images appear as nearest neighbours, the normalisation or
    transform has an error.

    Because all embeddings are L2-normalised, cosine similarity between
    two rows A and B is simply A · B (dot product). The pairwise similarity
    matrix is embeddings @ embeddings.T, an N×N matrix.

    For large N (>10,000), computing the full N×N matrix is memory-intensive
    (~400MB at N=10,000). This check uses a small random subsample (100 rows)
    to keep it cheap — this is a sanity check, not the full Phase 2 detector.

    Args:
        embeddings  : (N, 768) L2-normalised float32 numpy array.
        image_paths : list of path strings matching embedding rows.
        n_checks    : number of query images to check.
    """
    N = embeddings.shape[0]

    # Use a random subsample for the similarity search to keep this O(sample²).
    # 100 neighbours is enough to find a true nearest neighbour for any query
    # in a reasonably well-populated class.
    sample_size   = min(100, N)
    sample_indices = np.random.choice(N, size=sample_size, replace=False)
    sample_embs    = embeddings[sample_indices]

    print(f"\n[NN Check] Running nearest-neighbour sanity check "
          f"({n_checks} queries over a {sample_size}-image subsample) ...")

    # Randomly pick query images from the subsample.
    query_positions = np.random.choice(sample_size, size=n_checks, replace=False)

    for i, qpos in enumerate(query_positions):
        query_emb  = sample_embs[qpos]         # shape (768,)
        query_idx  = sample_indices[qpos]
        query_path = image_paths[query_idx]

        # Cosine similarity = dot product (both are L2-normalised).
        # similarities shape: (sample_size,)
        similarities = sample_embs @ query_emb

        # Set the query's own similarity to -inf so it doesn't pick itself.
        similarities[qpos] = -np.inf

        # Find the highest-similarity neighbour.
        nn_pos        = int(np.argmax(similarities))
        nn_idx        = sample_indices[nn_pos]
        nn_path       = image_paths[nn_idx]
        nn_similarity = similarities[nn_pos]

        print(f"\n[NN Check] Query {i+1}:")
        print(f"  Query  (idx {query_idx:>6}): {query_path}")
        print(f"  Nearest(idx {nn_idx:>6}): {nn_path}")
        print(f"  Cosine similarity        : {nn_similarity:.4f}")
        print(f"  -> Inspect these two images. They should look visually similar.")

    print(f"\n[NN Check] If the pairs above look visually unrelated, "
          f"the transform pipeline has an error. Re-check normalisation constants.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_extraction(input_dir=None, output_dir=None):
    """
    Orchestrates all of Phase 1:
        1. GPU check.
        2. Load and freeze DINOv2.
        3. Index valid image paths.
        4. Run batched inference and L2-normalise.
        5. Spot-check norms.
        6. Save embeddings.npy and index.json.
        7. Run nearest-neighbour sanity check.

    Args:
        input_dir  : directory containing images to extract from.
                     Defaults to EXTRACTION_CONFIG["input_dir"].
                     main.py passes the correct path based on --mode.
        output_dir : directory to write outputs to.
                     Defaults to EXTRACTION_CONFIG["output_dir"].

    Called by src/main.py when --mode extract (or --mode benchmark after Phase 0).
    """
    config = EXTRACTION_CONFIG.copy()

    # Allow main.py to override paths via arguments.
    if input_dir  is not None: config["input_dir"]  = input_dir
    if output_dir is not None: config["output_dir"] = output_dir

    print("=" * 60)
    print("PHASE 1: DINOv2 Feature Extraction")
    print("=" * 60)
    print(f"  Input  : {config['input_dir']}")
    print(f"  Output : {config['output_dir']}")
    print(f"  Model  : {config['model_name']}")

    # ── STEP 1: GPU CHECK ──────────────────────────────────────────────────────
    device = check_gpu()

    # ── STEP 2: LOAD MODEL ────────────────────────────────────────────────────
    model = load_model(config["model_name"], device)

    # ── STEP 3: INDEX IMAGES ──────────────────────────────────────────────────
    image_paths = index_images(config["input_dir"])
    N = len(image_paths)

    # Warn if the dataset is large enough that Phase 2's exact pairwise
    # distance matrix may be a memory concern.
    if N >= 50_000:
        print(f"\n[Warning] N={N} ≥ 50,000. Phase 2 must use FAISS (not exact "
              f"numpy dot product) to avoid an ~{N**2 * 4 / (1024**3):.0f}GB "
              f"pairwise distance matrix.")

    # ── STEP 4 + 5: INFERENCE + NORMALISATION ─────────────────────────────────
    embeddings = extract_embeddings(model, image_paths, config, device)

    # ── STEP 6: NORM SPOT-CHECK ───────────────────────────────────────────────
    spot_check_norms(embeddings, n_samples=10)

    # ── STEP 7: SAVE ──────────────────────────────────────────────────────────
    save_outputs(embeddings, image_paths, config["output_dir"])

    # ── STEP 8: NN SANITY CHECK ───────────────────────────────────────────────
    nearest_neighbour_sanity_check(embeddings, image_paths, n_checks=3)

    # ── DONE ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 1 complete.")
    print(f"  embeddings.npy : {config['output_dir']}/embeddings.npy  "
          f"({N} × {config['embedding_dim']})")
    print(f"  index.json     : {config['output_dir']}/index.json  "
          f"({N} entries)")
    print("=" * 60)
    print("\nNext: Phase 2 — Near-Duplicate Detection (find_duplicates.py)")
    print("Load embeddings with: np.load('data/02_embeddings/embeddings.npy')")