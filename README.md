# Uncertainty Estimation in Fine-Tuned Language Models

Code accompanying the Master's thesis *"Uncertainty Estimation in Fine-Tuned Language Models: A Comparative Study of Bayesian Deep Learning Methods"* (Niklas Thomann, LMU Munich, Department of Statistics — M.Sc. Statistics & Data Science, supervised by Prof. Dr. David Rügamer).

The thesis compares five approaches to uncertainty quantification for a LoRA-fine-tuned Qwen2.5-0.5B model on AG News text classification:

- **MILE** — Microcanonical Langevin Ensembles (full-batch MCLMC)
- **pSMILE** — a mini-batch variant of MILE
- **Mean-field Variational Inference** (Bayes-by-Backprop)
- **Laplace approximation** (diagonal empirical Fisher)
- **Deep Ensembles**

against a deterministic point-estimate baseline, evaluated on predictive performance, calibration, entropy-based uncertainty decomposition, and out-of-distribution detection (far-OOD: IMDB, near-OOD: HuffPost).

## Repository Structure

```
.
├── notebooks/          # Experiment notebooks, organized by method (see below)
├── src/
│   └── qwen_posterior_utils.py   # Shared, method-agnostic posterior/eval utilities
├── configs/             # Early Wandb sweep configs (superseded by the notebook-embedded
│                        # configuration used for the final experiments; kept for history)
├── results/             # Early proof-of-concept run notes (see note below)
└── data/                # Empty on purpose — see "Data" section
```

### `notebooks/`

| Folder | Contents |
|---|---|
| `01_setup/` | Baseline LoRA fine-tuning setup and Deep Ensemble member training |
| `02_mile/` | MILE (full-batch MCLMC, multi-chain) and, in `02_mile/SMILE/`, the pSMILE step-size search (coarse grid → three fine candidates `1e-4`/`3e-4`/`5e-4` → `1e-4` adopted) |
| `03_mfvi/` | Mean-field variational inference, incl. the KL-weight validation sweep |
| `04_laplace/` | Diagonal Laplace approximation |
| `05_deep_ensemble/` | Deep Ensemble training/metrics |
| `06_evaluation/` | Combined evaluation, far-/near-OOD detection |

A few legacy notebooks from earlier iterations of the pipeline (e.g. `02_methods_v3_2.ipynb`, `02_mile_warmup_extended.ipynb`, `Evaluation.ipynb`) still sit at the top level of `notebooks/` pending cleanup — they document intermediate diagnostic steps (e.g. the MILE chain-instability investigation at $N_{\text{PER\_CLASS}}=512$ described in the thesis) but are not required to reproduce the final reported numbers.

## Dependencies

This project builds on two external, actively-developed research codebases rather than reimplementing model training and MCLMC sampling from scratch:

| Dependency | Role | Repository | Used at commit |
|---|---|---|---|
| **subspace_inference** (`bayes_sub_inf`) | Bézier-curve/LoRA subspace training framework (Daniel Dold) | Fork with thesis-specific extensions (Qwen posterior utilities, AG News experiment scripts): [`NiklasThom/bayes_sub_inf`](https://github.com/NiklasThom/bayes_sub_inf) (forked from [`doldd/bayes_sub_inf`](https://github.com/doldd/bayes_sub_inf), MIT license) | see fork's `main` branch |
| **MILE** | Microcanonical Langevin Ensembles reference implementation (Sommer et al., ICLR 2025) | [`EmanuelSommer/MILE`](https://github.com/EmanuelSommer/MILE) — used unmodified | `2c6dd8d170d991c5c2884153aa8a1e26297cbaca` |
| **SMILE** | Mini-batch MCLMC reference implementation (Sommer et al.) | [`EmanuelSommer/SMILE`](https://github.com/EmanuelSommer/SMILE) — used unmodified | `8a54636ac514488ec8d900436b5b4aa7e9faf85d` |

The Python environment (JAX, Flax, Optax, BlackJAX, Transformers, ...) is managed entirely through `bayes_sub_inf`'s own `uv`-based setup — see [Setup](#setup) below. `requirements.txt` in this repo mirrors those same pinned versions for reference / non-`uv` workflows, but is **not** the authoritative source (that is `bayes_sub_inf`'s `pyproject.toml` / `uv.lock`).

## Setup

```bash
# 1. Clone this repo and the three dependencies alongside each other
git clone https://github.com/NiklasThom/Masterarbeit.git
git clone https://github.com/NiklasThom/bayes_sub_inf.git
git clone https://github.com/EmanuelSommer/MILE.git && (cd MILE && git checkout 2c6dd8d170d991c5c2884153aa8a1e26297cbaca)
git clone https://github.com/EmanuelSommer/SMILE.git && (cd SMILE && git checkout 8a54636ac514488ec8d900436b5b4aa7e9faf85d)

# 2. Install the environment (from inside bayes_sub_inf)
cd bayes_sub_inf
uv sync --extra viz

# 3. Place src/qwen_posterior_utils.py from this repo at the root of bayes_sub_inf
#    (notebooks add bayes_sub_inf to sys.path and import it directly)
cp ../Masterarbeit/src/qwen_posterior_utils.py .

# 4. Run a notebook (example: Laplace)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_laplace/04_laplace.ipynb
```

Notebooks read their experiment-specific parameters from environment variables (with sensible defaults), so the same notebook is reused across configurations, e.g.:

```bash
export N_PER_CLASS=2048   # posterior-fitting subset size (x4 classes)
export SEED=1
export N_SAMPLES=30       # posterior draws for evaluation
```

## Data

`data/` is intentionally empty (only tracked via `.gitkeep`) — the tokenized AG News dataset (`train_data.npz` / `test_data.npz`, ~1 GB) is regenerated, not committed. From within `bayes_sub_inf`:

```bash
python tests/qwen_data/helper_scripts/save_datasets_and_params.py --mode all
```

This uses a `max_seq_len=128` tokenization for AG News (confirmed directly in that script's dataset configuration).

## Results

`results/README.md` currently only documents an early proof-of-concept run (2026-05-24, 1,000 examples, ~0.66 accuracy) from before the pipeline reached its final form — it predates and does **not** reflect the numbers reported in the thesis. The authoritative results are the thesis PDF itself (Chapter 6) together with the `method_results/` JSON summaries produced by each notebook on the cluster.

## Reproducing specific thesis results

| Thesis item | Source |
|---|---|
| Table 4 / 8 — MILE ($N_{\text{PER\_CLASS}}=2048$, $K=3$, repaired) | `02_mile/` multi-chain run + chain-1 repair, final pooled evaluation |
| Table 4 / 8 — pSMILE | `02_mile/SMILE/02_mile_psmile_prod_step1e-4.ipynb` |
| Table 4 / 8 — MFVI | `03_mfvi/03_mfvi.ipynb` |
| Table 4 / 8 — Laplace | `04_laplace/04_laplace.ipynb` |
| Table 4 / 8 — Deep Ensemble | `05_deep_ensemble/` |
| Tables 9–10, Figure 5 — OOD (far: IMDB, near: HuffPost) | `06_evaluation/ood_evaluation.ipynb`, `06_evaluation/near_ood_evaluation.ipynb` |
| Table 11, Figure 6 — computational cost | timing fields in each method's saved run summary (`method_results/<method>/*_summary.json`) |

## Citation

If you build on this work, please cite the thesis and, where relevant, the underlying MILE/SMILE papers this implementation is based on (Sommer et al.).
