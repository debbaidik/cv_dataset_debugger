# CV Dataset Debugger

A statistical pipeline for auditing computer vision datasets. Translates raw images into a structured mathematical space and runs a sequence of tests to surface near-duplicates, structural outliers, mislabeled samples, and systemic collection bias — with measurable precision and recall against a synthetic ground truth.

---

## The Problem

Every CV model is only as good as the data it trains on. Most dataset auditing is manual: a human scrolls through images and flags problems by eye. This does not scale, and it finds the obvious errors while missing the subtle ones — the image that looks normal but carries the wrong label, the corrupted file that passes a filename check, the cluster of mislabeled samples that all came from the same broken data source.

This pipeline treats data quality as a statistical problem. Images are embedded into a high-dimensional vector space using a pretrained Vision Transformer, and then five phases of mathematical analysis find what the human eye cannot.

---

## Architecture

```
Raw Images → DINOv2 Embeddings → Near-Duplicate Filter → Outlier Filter → Label Error Detection → Bias Clustering
```

Each phase operates on the same N×768 feature matrix produced in Phase 1. The heavy inference runs once; all statistical tests run on the saved matrix.

---

## The Math

### Phase 0 — Synthetic Benchmark

Before the pipeline runs on real data, a corruption engine injects three types of synthetic errors into a clean base dataset at configurable rates:

| Corruption Type  | Default Rate | What It Mimics |
|-----------------|--------------|----------------|
| Label noise     | 15%          | Human annotator error |
| Image corruption| 5%           | Failed download, broken sensor |
| Near-duplicates | 3%           | Video frame sampling, re-uploads |

Every injected error is logged to `ground_truth.json` with its index, type, original label, and injected label. This manifest is the ruler against which all downstream phases are scored.

### Phase 1 — Vectorization (DINOv2)

Images are batched through the frozen backbone of `dinov2_vitb14`, a self-supervised Vision Transformer pretrained by Meta AI. The classification head is discarded; the 768-dimensional `[CLS]` token output of the penultimate layer is extracted for each image.

Every output vector is L₂ normalized onto the unit hypersphere:

$$v_{\text{norm}} = \frac{v}{\|v\|_2}$$

This ensures all downstream distance calculations operate on semantic angle rather than vector magnitude. The result is an N×768 feature matrix saved to disk. The images are now numbers.

**Model choice rationale:** `dinov2_vitb14` (Base variant, 768 dimensions) is chosen over the Small variant (384 dimensions) for covariance matrix stability, and over Large/Giant variants (1024/1536 dimensions) to avoid the curse of dimensionality degrading cosine discriminability. For a data auditing task, Base representations are sufficient.

### Phase 2 — Epsilon Filter (Near-Duplicates)

Pairwise cosine similarity is computed across all embedding vectors. A threshold ε defines the near-duplicate boundary: if the similarity between vector A and vector B exceeds 1 − ε, they are flagged as near-duplicates.

For datasets under ~50,000 images, exact pairwise search is used. Above that threshold, FAISS Approximate Nearest Neighbours (ANN) is introduced as an explicit architectural decision with documented precision tradeoffs.

**Evaluation:** Precision and recall are computed against `ground_truth.json` as ε is swept from strict to loose, producing a full PR curve. The operating threshold is selected at the point of maximum F1.

### Phase 3 — Covariance Filter (Structural Outliers)

Out-of-distribution samples — corrupted files, images from the wrong domain — are identified using Mahalanobis distance, which accounts for the variance and directional shape of the embedding distribution rather than treating all dimensions equally.

The centroid μ and covariance matrix Σ of all embeddings are computed. For each embedding v, the Mahalanobis distance is:

$$D_M(v) = \sqrt{(v - \mu)^T \Sigma^{-1} (v - \mu)}$$

Embeddings at the extreme statistical tails of this distribution are flagged as structural outliers. The tail threshold is a configurable percentile cutoff.

**Evaluation:** Injected image corruptions (Gaussian noise) should appear at the far tail of the Mahalanobis distribution. Recovery is scored against the ground truth manifest.

### Phase 4 — Joint Probability Matrix (Mislabeled Data)

Label errors are detected using Confident Learning. A cross-validated linear classifier is trained on the clean embeddings, producing a softmax probability distribution over all classes for every image.

A joint probability matrix C is constructed where each entry C[i][j] represents the estimated probability that an image has given label i but true label j. Images where the predicted probability of their given label is substantially lower than the predicted probability of an alternative label are isolated as label errors.

**Evaluation:** Injected label noise is the target positive class. The F1 score at the chosen operating threshold is the primary reported metric for this phase.

### Phase 5 — Bias Discovery (Clustering)

Flagged errors from Phases 3 and 4 are clustered using DBSCAN to test the hypothesis that errors are not randomly distributed in embedding space.

**Hypothesis:** *Errors flagged by this pipeline are not randomly distributed. They form statistically significant clusters, indicating systemic rather than random data collection flaws.*

**Test:** The mean intra-cluster distance of flagged points is compared against the mean intra-cluster distance of an equal-sized random sample from clean points. If flagged errors cluster more tightly, the hypothesis is supported — the errors are systemic, not random.

The finding is reported as a specific, testable claim, not a visualization.

---

## Results

*Populated after benchmark run. Target metrics at chosen operating thresholds:*

| Phase | Task | Precision | Recall | F1 |
|-------|------|-----------|--------|----|
| 2 | Near-duplicate detection | — | — | — |
| 3 | Structural outlier detection | — | — | — |
| 4 | Label error detection | — | — | — |

*Benchmark: CIFAR-10, 15% label noise + 5% image corruption + 3% duplicates injected. Seed: 42.*

---

## Filesystem

```
cv_dataset_debugger/
│
├── data/
│   ├── 01_raw/                  ← Original, untouched images
│   ├── 02_embeddings/           ← DINOv2 feature matrix (.npy) + image index
│   ├── 03_reports/              ← Flagged index CSVs from each phase
│   └── 04_benchmark/
│       ├── ground_truth.json    ← Immutable corruption manifest
│       └── corrupted/           ← Synthetically corrupted dataset copy
│
├── notebooks/
│   ├── 00_benchmark_design.ipynb   ← Visual verification of injected corruptions
│   ├── 01_eda_embeddings.ipynb     ← Latent space exploration
│   ├── 02_distance_testing.ipynb   ← Threshold sensitivity analysis
│   ├── 03_error_clusters.ipynb     ← DBSCAN cluster visualisation
│   └── 04_benchmark_eval.ipynb     ← PR curves and F1 scores
│
├── src/
│   ├── benchmark/
│   │   ├── corrupt_dataset.py   ← Injection engine and manifest writer
│   │   └── evaluate.py          ← Precision, Recall, F1, PR curve builder
│   ├── extraction/
│   │   └── dinov2_extractor.py  ← Batched inference, L₂ normalisation, save
│   ├── core_math/
│   │   ├── distances.py         ← Mahalanobis, cosine similarity
│   │   └── distributions.py     ← Covariance estimators
│   ├── analysis/
│   │   ├── find_duplicates.py   ← ANN + epsilon threshold
│   │   ├── find_outliers.py     ← Mahalanobis tail test
│   │   ├── find_mislabeled.py   ← Confident Learning matrix
│   │   └── cluster_errors.py    ← DBSCAN on flagged subsets
│   └── main.py                  ← Pipeline orchestrator
│
├── requirements.txt
└── README.md
```

---

## Setup

**Requirements:** Python 3.10+, CUDA-enabled GPU (tested on RTX 3050 6GB)

```bash
# 1. Clone and enter the project
git clone https://github.com/your-username/cv-dataset-debugger
cd cv_dataset_debugger

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate        # Windows
source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify GPU visibility
python -c "import torch; print(torch.cuda.is_available())"
# Must return True before proceeding
```

---

## Usage

```bash
# Full pipeline in benchmark mode (recommended first run)
# Builds corrupted dataset, extracts embeddings, runs all phases, evaluates
python -m src.main --mode benchmark

# Skip Phase 0 if benchmark already exists
python -m src.main --mode benchmark --skip-corruption

# Production mode — run on raw data without evaluation
python -m src.main --mode production
```

All commands must be run from the project root directory (`cv_dataset_debugger/`), not from inside `src/`.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` + `torchvision` | DINOv2 inference |
| `numpy` + `scipy` | Matrix operations, covariance, Mahalanobis |
| `scikit-learn` | Cross-validated classifier, DBSCAN, K-means |
| `faiss-cpu` | Approximate nearest neighbours (large datasets) |
| `cleanlab` | Confident Learning implementation |
| `Pillow` | Image loading and corruption injection |
| `matplotlib` + `seaborn` | PR curves and latent space visualisation |

---

## Key Design Decisions

**Why DINOv2 and not a supervised model?** Self-supervised pretraining means the backbone generalises to domain-specific imagery (medical, satellite, industrial) without fine-tuning. A supervised ImageNet classifier would embed images based on ImageNet categories, which may be irrelevant to your dataset's semantics.

**Why L₂ normalisation?** Projecting embeddings onto the unit hypersphere decouples distance calculations from vector magnitude. Two images that are semantically identical but differ in brightness or contrast produce vectors of different magnitudes but similar directions. After normalisation, cosine similarity captures that correctly.

**Why build the benchmark before the pipeline?** Building the ground truth first forces honest threshold selection. Retrofitting a benchmark after the pipeline is tuned produces inflated metrics.

**Why DBSCAN over K-means for Phase 5?** K-means requires specifying the number of clusters upfront, which assumes you know the structure of your errors. DBSCAN discovers cluster count from data density, which is more appropriate when testing whether errors cluster at all.

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Synthetic benchmark construction | ✅ Complete |
| 1 | DINOv2 feature extraction | ✅ Complete |
| 2 | Near-duplicate detection | ✅ Complete |
| 3 | Structural outlier detection | ✅ Complete |
| 4 | Label error detection | ✅ Complete |
| 5 | Bias clustering | ✅ Complete |

