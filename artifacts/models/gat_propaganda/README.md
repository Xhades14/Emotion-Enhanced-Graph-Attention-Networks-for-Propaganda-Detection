# Graph Attention Network (GAT) for Propaganda Classification
select, rerank and then train meta-classifier
hindi emo model
tw-neg-358
## Overview

This directory contains the trained GAT model and learned graph artifacts for propaganda classification on social media conversations. The model uses Graph Attention Networks to leverage both tweet content (via sentence embeddings) and conversation structure to detect propaganda.

## Model Performance

### Test Set Results
- **Test Accuracy**: 81.9%
- **Test F1 Score**: 83.2%
- **Test AUC**: 86.2%
- **Optimal Threshold**: 0.55 (well-calibrated probabilities)

### Comparison with Baselines

| Model | Accuracy | F1 Score | AUC |
|-------|----------|----------|-----|
| **GAT (this model)** | **81.9%** | **83.2%** | **86.2%** |
| Meta-classifier | 67.0% | 65.0% | ~70.0% |
| Text-only baseline | 54.0% | 40.0% | ~55.0% |

**Improvements:**
- **+22% accuracy** over meta-classifier
- **+28% F1 score** over meta-classifier
- **+23% AUC** over meta-classifier

## Architecture Details

### Model Components

1. **Input Features** (769 dimensions):
   - 768-dim sentence embeddings (paraphrase-multilingual-mpnet-base-v2)
   - 1-dim meta-classifier propaganda score
   - Features independently normalized with StandardScaler

2. **GAT Architecture**:
   - 3 GATConv layers with `add_self_loops=False`
   - Hidden dimension: 128
   - Attention heads: 8 (layers 1-2), 1 (layer 3)
   - Dropout: 0.2
   - Activation: ELU

3. **Training Configuration**:
   - Loss: Focal Loss (α=0.75, γ=1.5)
   - Optimizer: Adam (lr=5e-3)
   - Early stopping: validation AUC (patience=50)
   - Gradient clipping: max_norm=5.0

### Graph Construction

**Semantic Similarity Graph:**
- **Method**: Cosine similarity between sentence embeddings
- **Parameters**: 
  - k=20 neighbors per node
  - Similarity threshold: 0.3
  - Bidirectional edges
- **Result**: 29,406 edges connecting 1,154 nodes

**Key Design Choice:** Self-loops explicitly disabled (`add_self_loops=False`) to force message passing through neighbors rather than node-wise classification.

## Training Dynamics

### Convergence
- Converged at epoch 199
- Best validation AUC: 84.5%
- Final training F1: 82.2%
- Final validation F1: 80.3%
- Minimal overfitting observed

### Probability Calibration
- Training probability mean: 0.581
- Validation probability mean: 0.574
- Well-calibrated around 0.5 (healthy distribution)
- No probability collapse observed

## Graph Analysis

### Overall Statistics
- **Nodes**: 1,154 tweets
- **Edges**: 29,406 connections
- **Average degree**: 50.96 neighbors per tweet
- **Mean attention weight**: 0.0392
- **Attention std**: 0.0500

### Hub Nodes (High-Degree Propaganda Centers)

| Node ID | Degree | Description |
|---------|--------|-------------|
| 759 | 200 | Most influential hub |
| 519 | 180 | Secondary hub |
| 125 | 176 | Secondary hub |
| 246 | 160 | Tertiary hub |
| 46 | 158 | Tertiary hub |

These hub nodes represent central propaganda themes or narratives that share semantic similarity with many other tweets in the dataset.

### Most Attended Edges

Top edges by learned attention weights (model considers these connections most important):

| Source → Target | Attention Weight | Interpretation |
|-----------------|------------------|----------------|
| 129 → 340 | 1.0000 | Perfect semantic match |
| 73 → 42 | 1.0000 | Perfect semantic match |
| 567 → 366 | 1.0000 | Perfect semantic match |
| 815 → 800 | 0.9490 | Very strong similarity |
| 752 → 726 | 0.8141 | Strong similarity |
| 627 → 1054 | 0.7917 | Strong similarity |
| 627 → 820 | 0.7917 | Strong similarity |

The model learned to assign maximum attention (1.0) to three tweet pairs that are semantically identical or near-identical, indicating successful learning of meaningful relationships.

### Attention Weight Distribution

- **Mean**: 0.0392 (most edges have low attention)
- **Std**: 0.0500 (moderate variance)
- **Min**: 0.0000 (some edges ignored)
- **Max**: 1.0000 (critical edges fully attended)

The distribution shows the model learned to **focus attention selectively** on the most relevant neighbors rather than treating all connections equally.

## Key Technical Insights

### What Made This Work

1. **Removing Self-Loops**: Setting `add_self_loops=False` in GATConv forced true neighborhood aggregation instead of node-wise classification.

2. **Focal Loss**: Using α=0.75, γ=1.5 prevented probability collapse and balanced class focus without training instability.

3. **Semantic Graph**: Building connections via embedding similarity (vs random) gave the model meaningful structure to learn from.

4. **Early Stopping on AUC**: Using validation AUC instead of F1 avoided stopping at degenerate states where F1=0 due to threshold mismatch.

5. **Independent Feature Scaling**: Normalizing embeddings and meta-scores separately prevented the 768-dim embedding from dominating the single meta-score.

### Common Pitfalls Avoided

❌ **Self-loops**: Default PyG behavior adds self-loops, causing models to ignore neighbors  
✅ **Solution**: Explicitly set `add_self_loops=False`

❌ **Probability collapse**: BCE with pos_weight pushed all predictions to extreme values  
✅ **Solution**: Focal loss with proper α, γ parameters

❌ **Threshold mismatch**: F1=0 during training with probabilities far from 0.5  
✅ **Solution**: Threshold calibration on validation set (found optimal=0.55)

❌ **Random graphs**: No semantic meaning in connections  
✅ **Solution**: Semantic similarity graph with cosine threshold

## Files in This Directory

- **`gat_model.pt`**: Trained model state dict (PyTorch checkpoint)
- **`learned_edges.csv`**: All 29,406 edges with attention weights
- **`learned_graph.gpickle`**: Full NetworkX graph (pickled)
- **`node_embeddings.pt`**: Node representations at each layer (input, hidden1, hidden2)
- **`learned_graph_visualization.png`**: Basic graph visualization (top 100 edges)
- **`graph_visualization_detailed.png`**: Comprehensive 4-panel visualization:
  - Spring layout (attention-weighted)
  - Circular layout
  - Degree distribution
  - Attention weight distribution

## Usage

### Load Trained Model

```python
import torch
from train_gat_propaganda import GATClassifier

# Load model
model = GATClassifier(in_channels=769, hidden_dim=128, heads=8, dropout=0.2)
model.load_state_dict(torch.load("artifacts/models/gat_propaganda/gat_model.pt"))
model.eval()
```

### Inference on Single Tweet

```python
# Run inference using the training script
python scripts/train_gat_propaganda.py --inference --tweet-idx 42

# Or programmatically:
from train_gat_propaganda import predict_single_tweet

result = predict_single_tweet(model, data, node_idx=42, threshold=0.55)
print(f"Probability: {result['probability']:.4f}")
print(f"Prediction: {result['label']}")
```

### Load and Analyze Graph

```python
import pickle
import pandas as pd

# Load edges
edges_df = pd.read_csv("artifacts/models/gat_propaganda/learned_edges.csv")

# Load graph
with open("artifacts/models/gat_propaganda/learned_graph.gpickle", "rb") as f:
    graph = pickle.load(f)

# Analyze attention
top_edges = edges_df.nlargest(10, "weight")
print(top_edges)
```

### Visualize Graph

```python
python scripts/visualize_gat_graph.py
```

This generates a detailed 4-panel visualization showing network structure, degree distribution, and attention patterns.

## Reproduction

To retrain the model with the same configuration:

```bash
python scripts/train_gat_propaganda.py \
  --use-semantic-graph \
  --neighbor-k 20 \
  --similarity-threshold 0.3 \
  --hidden-dim 128 \
  --heads 8 \
  --lr 5e-3 \
  --epochs 500 \
  --patience 50 \
  --seed 42
```

## Dataset

- **Source**: `artifacts/prop_datasets/tree_width/merged_conversations.jsonl`
- **Meta-features**: `artifacts/prop_datasets/tree_width/meta_features.csv`
- **Meta-classifier**: `artifacts/models/meta_classifier.joblib`
- **Total samples**: 1,154 tweets
- **Class balance**: ~50% propaganda, ~50% non-propaganda
- **Train/Val/Test split**: 80% / 10% / 10% (stratified)

## Future Improvements

1. **Cross-validation**: Test performance stability across different random seeds
2. **Hyperparameter tuning**: Grid search over k, threshold, hidden_dim, heads
3. **Temporal modeling**: Incorporate tweet timestamps for temporal graph evolution
4. **Multi-task learning**: Joint training for propaganda + emotion detection
5. **Explainability**: Extract and analyze high-attention subgraphs for interpretation
6. **Transfer learning**: Apply to other misinformation detection tasks

## References

- **Graph Attention Networks**: Veličković et al. (2017) - https://arxiv.org/abs/1710.10903
- **Focal Loss**: Lin et al. (2017) - https://arxiv.org/abs/1708.02002
- **Sentence-Transformers**: Reimers & Gurevych (2019) - https://arxiv.org/abs/1908.10084

---

**Training Date**: November 14, 2025  
**Model Version**: 1.0  
**Framework**: PyTorch 2.x + PyTorch Geometric
