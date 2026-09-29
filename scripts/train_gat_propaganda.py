"""Train a Graph Attention Network for propaganda classification."""
from __future__ import annotations

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

DEFAULT_DATA_PATH = Path("artifacts") / "prop_datasets" / "tree_width" / "merged_conversations.jsonl"
DEFAULT_META_FEATURES = Path("artifacts") / "prop_datasets" / "tree_width" / "meta_features.csv"
DEFAULT_META_MODEL = Path("artifacts") / "models" / "meta_classifier.joblib"
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_OUTPUT_DIR = Path("artifacts") / "models" / "gat_propaganda"

META_FEATURE_COLUMNS = [
    "text_pred",
    "mean_anger",
    "mean_disgust",
    "mean_fear",
    "mean_joy",
    "mean_sadness",
    "mean_surprise",
    "mean_neutral",
    "entropy",
    "variance",
]


@dataclass
class Config:
    data_path: Path
    meta_features_path: Path
    meta_model_path: Path
    embedding_model_name: str
    output_dir: Path
    device: str
    batch_size: int
    fully_connected: bool
    neighbor_k: int
    similarity_threshold: float
    use_semantic_graph: bool
    seed: int
    hidden_dim: int
    heads: int
    dropout: float
    lr: float
    epochs: int
    patience: int


class GATClassifier(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_dim, heads=heads, dropout=dropout, add_self_loops=False)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, add_self_loops=False)
        self.conv3 = GATConv(hidden_dim * heads, 1, heads=1, concat=False, dropout=dropout, add_self_loops=False)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tuple[Tensor, Tensor]]]:
        h1 = self.conv1(x, edge_index)
        h1 = F.elu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = self.conv2(h1, edge_index)
        h2 = F.elu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        if return_attention:
            logits, ( att_edge_index, att_weights ) = self.conv3(
                h2,
                edge_index,
                return_attention_weights=True,
            )
            return logits.view(-1), h1.detach(), h2.detach(), (att_edge_index, att_weights)

        logits = self.conv3(h2, edge_index)
        return logits.view(-1), h1.detach(), h2.detach(), None


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".jsonl", ".json"}:
        records: List[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        df = pd.DataFrame(records)
    else:
        raise ValueError("Unsupported dataset format; provide CSV or JSONL")

    rename_map = {}
    if "post_id" in df.columns and "tweet_id" not in df.columns:
        rename_map["post_id"] = "tweet_id"
    if "label" in df.columns and "true_label" not in df.columns:
        rename_map["label"] = "true_label"
    df = df.rename(columns=rename_map)

    required = {"tweet_id", "root_text", "true_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    df = df.sort_values("tweet_id").reset_index(drop=True)
    return df


def load_meta_score(df: pd.DataFrame, meta_features_path: Path, meta_model_path: Path) -> pd.Series:
    meta_df = pd.read_csv(meta_features_path)
    if "post_id" in meta_df.columns and "tweet_id" not in meta_df.columns:
        meta_df = meta_df.rename(columns={"post_id": "tweet_id"})

    missing = {"tweet_id", *META_FEATURE_COLUMNS} - set(meta_df.columns)
    if missing:
        raise ValueError(f"Meta feature file missing columns: {sorted(missing)}")

    meta_model = joblib.load(meta_model_path)
    merged = df.merge(meta_df[["tweet_id", *META_FEATURE_COLUMNS]], on="tweet_id", how="left")
    if merged[META_FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("Meta features contain nulls after merge; ensure alignment")

    scores = meta_model.predict_proba(merged[META_FEATURE_COLUMNS])[:, 1]
    return pd.Series(scores, index=df.index, name="prop_score")


def compute_embeddings(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def build_semantic_graph(embeddings: np.ndarray, k: int, threshold: float = 0.0) -> nx.DiGraph:
    """Build graph by connecting nodes with similar embeddings."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    num_nodes = embeddings.shape[0]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    
    similarity = cosine_similarity(embeddings)
    np.fill_diagonal(similarity, -np.inf)
    
    for src in range(num_nodes):
        top_k_indices = np.argsort(similarity[src])[-k:][::-1]
        for dst in top_k_indices:
            if similarity[src, dst] > threshold:
                graph.add_edge(src, dst)
                graph.add_edge(dst, src)
    
    return graph


def build_graph(num_nodes: int, fully_connected: bool, k: int, seed: int) -> nx.DiGraph:
    """Legacy random graph builder (kept for compatibility)."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    rng = np.random.default_rng(seed)

    if fully_connected or k <= 0 or k >= num_nodes:
        for src in range(num_nodes):
            for dst in range(num_nodes):
                if src != dst:
                    graph.add_edge(src, dst)
        return graph

    for src in range(num_nodes):
        candidates = [dst for dst in range(num_nodes) if dst != src]
        if not candidates:
            continue
        sample_size = min(k, len(candidates))
        neighbors = rng.choice(candidates, size=sample_size, replace=False)
        for dst in neighbors:
            graph.add_edge(src, dst)
            graph.add_edge(dst, src)
    return graph


def to_edge_index(graph: nx.DiGraph) -> Tensor:
    edges = list(graph.edges())
    if not edges:
        raise ValueError("Graph has no edges; increase connectivity or k")
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Explicitly remove any self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    
    if edge_index.size(1) == 0:
        raise ValueError("No edges remain after removing self-loops")
    
    return edge_index


def create_data_object(features: Tensor, labels: Tensor, edge_index: Tensor, seed: int) -> Data:
    from sklearn.model_selection import train_test_split
    
    num_nodes = features.size(0)
    indices = np.arange(num_nodes)
    y_numpy = labels.cpu().numpy().astype(int)
    
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=y_numpy
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed, stratify=y_numpy[temp_idx]
    )
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = Data(x=features, y=labels, edge_index=edge_index)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def compute_metrics(logits: Tensor, labels: Tensor, mask: Tensor) -> Tuple[float, float, float]:
    logits = logits[mask]
    labels = labels[mask]
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()

    y_true = labels.cpu().numpy().astype(int)
    y_pred = preds.cpu().numpy().astype(int)
    y_prob = probs.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return acc, f1, auc


def focal_loss(logits: Tensor, labels: Tensor, alpha: float = 0.75, gamma: float = 1.5) -> Tensor:
    """Focal loss to prevent prediction collapse.
    
    Args:
        alpha: Weight for positive class (0.75 = bias toward detecting propaganda)
        gamma: Focusing parameter (1.5 = moderate focus on hard examples)
    """
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    probs = torch.sigmoid(logits)
    pt = torch.where(labels == 1, probs, 1 - probs)
    focal_weight = (1 - pt) ** gamma
    alpha_weight = torch.where(labels == 1, alpha, 1 - alpha)
    loss = alpha_weight * focal_weight * bce_loss
    return loss.mean()


def train(
    model: GATClassifier,
    data: Data,
    config: Config,
) -> GATClassifier:
    optimizer = Adam(model.parameters(), lr=config.lr)

    best_state: Optional[dict] = None
    best_val_auc = float("-inf")
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        logits, _, _, _ = model(data.x, data.edge_index)
        loss = focal_loss(logits[data.train_mask], data.y[data.train_mask], alpha=0.75, gamma=1.5)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_eval, _, _, _ = model(data.x, data.edge_index)
            train_metrics = compute_metrics(logits_eval, data.y, data.train_mask)
            val_metrics = compute_metrics(logits_eval, data.y, data.val_mask)
            
            train_probs = torch.sigmoid(logits_eval[data.train_mask])
            val_probs = torch.sigmoid(logits_eval[data.val_mask])

        elapsed = time.time() - epoch_start
        
        if epoch % 10 == 0 or epoch < 5:
            logging.info(
                "epoch=%d loss=%.4f train_acc=%.3f train_f1=%.3f train_auc=%.3f val_acc=%.3f val_f1=%.3f val_auc=%.3f train_prob[mean=%.3f,std=%.3f] val_prob[mean=%.3f,std=%.3f] time=%.2fs",
                epoch,
                loss.item(),
                train_metrics[0],
                train_metrics[1],
                train_metrics[2],
                val_metrics[0],
                val_metrics[1],
                val_metrics[2],
                train_probs.mean().item(),
                train_probs.std().item(),
                val_probs.mean().item(),
                val_probs.std().item(),
                elapsed,
            )
        elif val_metrics[1] > 0.0:
            logging.info(
                "epoch=%d loss=%.4f train_f1=%.3f val_f1=%.3f val_auc=%.3f train_prob_mean=%.3f val_prob_mean=%.3f",
                epoch,
                loss.item(),
                train_metrics[1],
                val_metrics[1],
                val_metrics[2],
                train_probs.mean().item(),
                val_probs.mean().item(),
            )

        val_auc = val_metrics[2]
        if np.isnan(val_auc):
            val_auc = 0.0
        
        # Early exit if probabilities collapse
        if epoch > 20 and train_probs.mean().item() < 0.01 and train_probs.std().item() < 0.01:
            logging.warning("Probability collapse detected at epoch %d, stopping early", epoch)
            logging.warning("  train_prob: mean=%.4f, std=%.4f", train_probs.mean().item(), train_probs.std().item())
            break

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logging.info("Early stopping triggered at epoch %d (best_val_auc=%.3f)", epoch, best_val_auc)
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state")

    model.load_state_dict(best_state)
    return model


def predict_single_tweet(
    model: GATClassifier,
    data: Data,
    node_idx: int,
    threshold: float = 0.5,
) -> dict:
    """Run inference on a single tweet node."""
    model.eval()
    with torch.no_grad():
        logits, _, _, _ = model(data.x, data.edge_index)
        prob = torch.sigmoid(logits[node_idx]).item()
        pred = int(prob >= threshold)
    
    return {
        "node_idx": node_idx,
        "probability": prob,
        "prediction": pred,
        "label": "propaganda" if pred == 1 else "non-propaganda",
    }


def save_artifacts(
    output_dir: Path,
    model: GATClassifier,
    attention: Tuple[Tensor, Tensor],
    data: Data,
    hidden1: Tensor,
    hidden2: Tensor,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "gat_model.pt")

    edge_index, att_weights = attention
    weights = att_weights.mean(dim=1).cpu().numpy()
    sources = edge_index[0].cpu().numpy()
    targets = edge_index[1].cpu().numpy()
    edges = pd.DataFrame({"source": sources, "target": targets, "weight": weights})
    edges.to_csv(output_dir / "learned_edges.csv", index=False)

    graph = nx.DiGraph()
    graph.add_nodes_from(range(data.num_nodes))
    for src, dst, weight in zip(sources, targets, weights):
        graph.add_edge(int(src), int(dst), weight=float(weight))
    import pickle
    with open(output_dir / "learned_graph.gpickle", "wb") as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        top_edges = sorted(
            [(int(s), int(t), float(w)) for s, t, w in zip(sources, targets, weights)],
            key=lambda x: x[2],
            reverse=True
        )[:100]
        
        vis_graph = nx.DiGraph()
        vis_graph.add_nodes_from(range(min(50, data.num_nodes)))
        for src, dst, weight in top_edges:
            if src < 50 and dst < 50:
                vis_graph.add_edge(src, dst, weight=weight)
        
        if vis_graph.number_of_edges() > 0:
            pos = nx.spring_layout(vis_graph, seed=42, k=0.5, iterations=50)
            edge_weights = [vis_graph[u][v]['weight'] for u, v in vis_graph.edges()]
            
            plt.figure(figsize=(12, 10))
            nx.draw_networkx_nodes(vis_graph, pos, node_size=300, node_color='lightblue', alpha=0.8)
            nx.draw_networkx_edges(
                vis_graph, pos, width=[w * 3 for w in edge_weights],
                alpha=0.6, edge_color=edge_weights, edge_cmap=plt.cm.viridis,
                arrows=True, arrowsize=10, connectionstyle='arc3,rad=0.1'
            )
            nx.draw_networkx_labels(vis_graph, pos, font_size=8)
            plt.title("Learned Graph (Top 100 edges, first 50 nodes)", fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / "learned_graph_visualization.png", dpi=150, bbox_inches='tight')
            plt.close()
    except Exception as e:
        import logging
        logging.warning("Failed to generate graph visualization: %s", e)

    torch.save(
        {
            "input": data.x.cpu(),
            "hidden_layer1": hidden1.cpu(),
            "hidden_layer2": hidden2.cpu(),
        },
        output_dir / "node_embeddings.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GAT model for propaganda classification")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--meta-features", type=Path, default=DEFAULT_META_FEATURES)
    parser.add_argument("--meta-model", type=Path, default=DEFAULT_META_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fully-connected", action="store_true")
    parser.add_argument("--neighbor-k", type=int, default=50)
    parser.add_argument("--similarity-threshold", type=float, default=0.3)
    parser.add_argument("--use-semantic-graph", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--inference", action="store_true", help="Run inference mode on trained model")
    parser.add_argument("--tweet-idx", type=int, help="Node index for inference (0-based)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = Config(
        data_path=args.data,
        meta_features_path=args.meta_features,
        meta_model_path=args.meta_model,
        embedding_model_name=args.embedding_model,
        output_dir=args.output,
        device=args.device,
        batch_size=args.batch_size,
        fully_connected=args.fully_connected,
        neighbor_k=args.neighbor_k,
        similarity_threshold=args.similarity_threshold,
        use_semantic_graph=args.use_semantic_graph,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
    )

    logging.info("Loading dataset from %s", config.data_path)
    set_random_seed(config.seed)
    df = load_dataset(config.data_path)

    logging.info("Computing meta-model scores from %s", config.meta_model_path)
    prop_scores = load_meta_score(df, config.meta_features_path, config.meta_model_path)

    logging.info("Generating sentence embeddings using %s", config.embedding_model_name)
    embeddings = compute_embeddings(df["root_text"].tolist(), config.embedding_model_name, config.batch_size)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    prop_scores_scaled = scaler.fit_transform(prop_scores.to_numpy().reshape(-1, 1))
    
    features = np.concatenate([embeddings_scaled, prop_scores_scaled], axis=1)
    x = torch.from_numpy(features).float()
    y = torch.from_numpy(df["true_label"].astype(int).to_numpy()).float()

    logging.info("Constructing graph structure (semantic=%s)", config.use_semantic_graph)
    if config.use_semantic_graph:
        graph = build_semantic_graph(embeddings, config.neighbor_k, config.similarity_threshold)
    else:
        graph = build_graph(len(df), config.fully_connected, config.neighbor_k, config.seed)
    edge_index = to_edge_index(graph)
    logging.info("Graph has %d nodes and %d edges", graph.number_of_nodes(), graph.number_of_edges())

    data = create_data_object(x, y, edge_index, config.seed)
    data = data.to(config.device)

    model = GATClassifier(x.size(1), config.hidden_dim, config.heads, config.dropout).to(config.device)

    logging.info("Starting training")
    trained_model = train(model, data, config)

    trained_model.eval()
    with torch.no_grad():
        logits, hidden1, hidden2, attention = trained_model(
            data.x,
            data.edge_index,
            return_attention=True,
        )
        
        val_logits = logits[data.val_mask]
        val_labels = data.y[data.val_mask]
        val_probs = torch.sigmoid(val_logits)
        
        best_threshold = 0.5
        best_f1_for_threshold = 0.0
        for threshold in np.arange(0.1, 0.9, 0.05):
            preds = (val_probs >= threshold).long()
            f1 = f1_score(val_labels.cpu().numpy().astype(int), preds.cpu().numpy().astype(int))
            if f1 > best_f1_for_threshold:
                best_f1_for_threshold = f1
                best_threshold = threshold
        
        logging.info("Best validation threshold: %.2f (F1=%.3f)", best_threshold, best_f1_for_threshold)
        
        test_logits = logits[data.test_mask]
        test_labels = data.y[data.test_mask]
        test_probs = torch.sigmoid(test_logits)
        test_preds = (test_probs >= best_threshold).long()
        
        test_acc = accuracy_score(test_labels.cpu().numpy().astype(int), test_preds.cpu().numpy().astype(int))
        test_f1 = f1_score(test_labels.cpu().numpy().astype(int), test_preds.cpu().numpy().astype(int))
        try:
            test_auc = roc_auc_score(test_labels.cpu().numpy(), test_probs.cpu().numpy())
        except ValueError:
            test_auc = float('nan')
        
        logging.info(
            "test_acc=%.3f test_f1=%.3f test_auc=%.3f (threshold=%.2f)",
            test_acc,
            test_f1,
            test_auc,
            best_threshold,
        )

    if attention is None:
        raise RuntimeError("Attention weights not available")

    logging.info("Saving artifacts to %s", config.output_dir)
    save_artifacts(config.output_dir, trained_model, attention, data, hidden1, hidden2)

    edge_index, att_weights = attention
    weights = att_weights.mean(dim=1).cpu().numpy()
    sources = edge_index[0].cpu().numpy()
    targets = edge_index[1].cpu().numpy()
    
    logging.info("Learned graph statistics:")
    logging.info("  Total edges: %d", len(sources))
    logging.info("  Mean attention weight: %.4f", weights.mean())
    logging.info("  Std attention weight: %.4f", weights.std())
    logging.info("  Min attention weight: %.4f", weights.min())
    logging.info("  Max attention weight: %.4f", weights.max())
    
    top_k = min(10, len(weights))
    top_indices = np.argsort(weights)[-top_k:][::-1]
    logging.info("Top %d edges by attention weight:", top_k)
    for idx in top_indices:
        logging.info("  %d -> %d: %.4f", sources[idx], targets[idx], weights[idx])

    logging.info("Training complete")
    
    # Inference mode
    if args.inference and args.tweet_idx is not None:
        if args.tweet_idx < 0 or args.tweet_idx >= data.x.size(0):
            logging.error("Tweet index %d out of range [0, %d)", args.tweet_idx, data.x.size(0))
        else:
            result = predict_single_tweet(trained_model, data, args.tweet_idx, threshold=best_threshold)
            logging.info("\n=== Inference Result for Node %d ===", args.tweet_idx)
            logging.info("  Probability: %.4f", result["probability"])
            logging.info("  Prediction: %s", result["label"])
            logging.info("  Ground truth: %s", "propaganda" if data.y[args.tweet_idx].item() == 1 else "non-propaganda")
            logging.info("  Tweet: %s", df.iloc[args.tweet_idx]["root_text"][:100])


if __name__ == "__main__":
    main()
