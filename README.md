# Evidence Retrieval & Propaganda Detection Pipeline

A three-stage misinformation analysis pipeline with a Streamlit UI:

1. **Claim verification**: extracts factual claims, retrieves evidence with Gemini search and returns a verdict.
2. **Propaganda + emotion scoring**: English/Hindi propaganda classifiers, GoEmotions reply emotions and a meta-classifier.
3. **Graph attention network (GAT)**: refines the decision using a graph of 1,154 historical conversations.

## Setup

Requires Python 3.10–3.12 (tested on 3.11). A GPU is optional.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `torch-geometric` fails to install, follow the [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your PyTorch/CUDA version.

## Download models and data

The trained models and datasets are too large for git. Download the zip from:

**➡️ [DRIVE_LINK_PLACEHOLDER](DRIVE_LINK_PLACEHOLDER)**

Unzip it **into the `artifacts/` folder** (merge with what is already there). You should end up with:

```text
Project_Propaganda Detection in Social Media/
├── app_complete.py
└── artifacts/
    ├── demo/                                        # already in the repo
    ├── models/
    │   ├── gat_propaganda/                          # already in the repo
    │   ├── eng_prop_model/SemEval_Trained_Intermediate(final)/
    │   ├── hprop-lora-adapter/
    │   ├── eng_emo_model/models/
    │   └── meta_classifier.joblib
    └── prop_datasets/tree_width/merged_conversations.jsonl
```

The sentence embedding model (`paraphrase-multilingual-mpnet-base-v2`) is downloaded automatically from Hugging Face on first run.

## Gemini API key (optional)

```powershell
$env:GEMINI_API_KEY = "your_key"
```

Without the key the app still runs: it skips Stage 1 (claim verification) and goes straight to the propaganda detection stages (2 and 3). Never commit keys.

## Run

Run these commands from the repository root:

```powershell
streamlit run app_complete.py   # full 3-stage UI
streamlit run app.py            # Stage 1 only (needs GEMINI_API_KEY, no downloads)
pytest                          # tests
```

## Repository layout

| Path | Contents |
| --- | --- |
| `app_complete.py` | Full Streamlit UI (all three stages) |
| `app.py` | Stage 1 Streamlit UI |
| `src/` | Claim verification graph, agents, API clients |
| `scripts/` | Training, evaluation and dataset tools |
| `artifacts/models/` | All models; the GAT (`gat_propaganda/`) is committed, the rest come from the zip |
| `artifacts/prop_datasets/` | Datasets from the zip |
| `docs/` | Decision rules, guardrails, technical notes |
| `tests/` | Stage 1 smoke test |

## Results

| Model | Accuracy | F1 | AUC |
| --- | --- | --- | --- |
| Text-only baseline | 54% | 40% | ~55% |
| Meta-classifier (Stage 2) | 66% | 65% | ~70% |
| **GAT (Stage 3)** | **81.9%** | **83.2%** | **86.2%** |

See [artifacts/models/gat_propaganda/README.md](artifacts/models/gat_propaganda/README.md) for GAT training details and [docs/](docs/) for the full technical write-up.
