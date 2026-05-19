# CV Dataset Debugger

A statistical pipeline for auditing computer vision datasets. Embeds images into a high-dimensional vector space using DINOv2 and runs automated tests to find near-duplicates, outliers, label errors, and collection bias.

## How It Works

```
Raw Images -> DINOv2 Embeddings -> Near-Duplicate Filter -> Outlier Filter -> Label Error Detection -> Bias Clustering
```

| Phase | What It Does |
|-------|-------------|
| 0 | Injects synthetic errors (label noise, corruption, duplicates) to build a ground truth benchmark |
| 1 | Extracts 768-dim embeddings from DINOv2 ViT-B/14 and L2-normalizes them |
| 2 | Flags near-duplicates via cosine similarity with an epsilon threshold |
| 3 | Detects structural outliers using Mahalanobis distance from the embedding distribution |
| 4 | Identifies label errors using Confident Learning (cross-validated joint probability matrix) |
| 5 | Clusters flagged errors with DBSCAN to test whether errors are systemic or random |

## Setup

Requires Python 3.10+ and a CUDA GPU.

```bash
git clone https://github.com/BaidikDoesNotCode/cv_dataset_debugger.git
cd cv_dataset_debugger
python -m venv .venv
.venv\Scripts\Activate  # Windows (use source .venv/bin/activate on Linux/Mac)

# Install PyTorch for your CUDA version first:
# https://pytorch.org/get-started/locally/

pip install -r requirements.txt
```

## Usage

```bash
# Full pipeline with benchmark (recommended first run)
python -m src.main --mode benchmark

# Skip corruption if benchmark already built
python -m src.main --mode benchmark --skip-corruption

# Run on your own data without evaluation
python -m src.main --mode production
```

Run all commands from the project root.

## Project Structure

```
src/
  benchmark/       - Corruption injection and evaluation scoring
  extraction/      - DINOv2 batched inference and embedding export
  core_math/       - Mahalanobis distance, cosine similarity, covariance
  analysis/        - Duplicate finder, outlier detector, label checker, clustering
  main.py          - Pipeline orchestrator
notebooks/         - EDA, threshold analysis, PR curves, cluster visualization
data/              - Raw images, embeddings, reports (gitignored)
```

## Dependencies

torch, torchvision, numpy, scipy, scikit-learn, faiss-cpu, cleanlab, Pillow, matplotlib, seaborn
