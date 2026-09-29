"""Visualize the learned GAT graph with attention weights."""
import pickle
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load artifacts
artifact_dir = Path("artifacts/models/gat_propaganda")
edges_df = pd.read_csv(artifact_dir / "learned_edges.csv")

with open(artifact_dir / "learned_graph.gpickle", "rb") as f:
    graph = pickle.load(f)

print(f"Graph statistics:")
print(f"  Nodes: {graph.number_of_nodes()}")
print(f"  Edges: {graph.number_of_edges()}")
print(f"  Average degree: {sum(dict(graph.degree()).values()) / graph.number_of_nodes():.2f}")

# Get top edges by attention weight
top_k = 100
edges_df_sorted = edges_df.sort_values("weight", ascending=False).head(top_k)
print(f"\nTop {top_k} edges by attention weight:")
print(edges_df_sorted.head(10))

# Create subgraph with top edges
top_edges = [(int(row["source"]), int(row["target"])) for _, row in edges_df_sorted.iterrows()]
top_nodes = set()
for src, dst in top_edges:
    top_nodes.add(src)
    top_nodes.add(dst)

subgraph = graph.subgraph(list(top_nodes)[:50])  # Limit to first 50 nodes for visibility

print(f"\nSubgraph for visualization:")
print(f"  Nodes: {subgraph.number_of_nodes()}")
print(f"  Edges: {subgraph.number_of_edges()}")

# Visualization 1: Spring layout with edge weights
fig, axes = plt.subplots(2, 2, figsize=(18, 16))

# Plot 1: Spring layout
ax = axes[0, 0]
pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
edge_weights = [subgraph[u][v]['weight'] for u, v in subgraph.edges()]
nx.draw_networkx_nodes(subgraph, pos, node_size=400, node_color='lightblue', 
                        alpha=0.8, ax=ax)
edges = nx.draw_networkx_edges(subgraph, pos, width=[w * 3 for w in edge_weights],
                                alpha=0.6, edge_color=edge_weights, edge_cmap=plt.cm.viridis,
                                arrows=True, arrowsize=15, connectionstyle='arc3,rad=0.1', ax=ax)
nx.draw_networkx_labels(subgraph, pos, font_size=9, ax=ax)
ax.set_title(f"Spring Layout (Top {subgraph.number_of_nodes()} nodes)", fontsize=12)
ax.axis('off')
# Add colorbar using ScalarMappable
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
sm = ScalarMappable(cmap=plt.cm.viridis, norm=Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Attention Weight')

# Plot 2: Circular layout
ax = axes[0, 1]
pos_circular = nx.circular_layout(subgraph)
nx.draw_networkx_nodes(subgraph, pos_circular, node_size=400, node_color='lightcoral',
                       alpha=0.8, ax=ax)
nx.draw_networkx_edges(subgraph, pos_circular, width=[w * 3 for w in edge_weights],
                       alpha=0.5, edge_color=edge_weights, edge_cmap=plt.cm.plasma,
                       arrows=True, arrowsize=15, connectionstyle='arc3,rad=0.1', ax=ax)
nx.draw_networkx_labels(subgraph, pos_circular, font_size=9, ax=ax)
ax.set_title("Circular Layout", fontsize=12)
ax.axis('off')

# Plot 3: Degree distribution
ax = axes[1, 0]
degrees = [graph.degree(n) for n in graph.nodes()]
ax.hist(degrees, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Node Degree', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title(f'Degree Distribution (Full Graph)', fontsize=12)
ax.grid(alpha=0.3)
ax.text(0.7, 0.9, f'Mean: {np.mean(degrees):.1f}\nStd: {np.std(degrees):.1f}',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 4: Attention weight distribution
ax = axes[1, 1]
ax.hist(edges_df["weight"], bins=50, color='darkgreen', alpha=0.7, edgecolor='black')
ax.set_xlabel('Attention Weight', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Attention Weight Distribution (All Edges)', fontsize=12)
ax.grid(alpha=0.3)
ax.text(0.6, 0.9, f'Mean: {edges_df["weight"].mean():.4f}\nStd: {edges_df["weight"].std():.4f}\nMax: {edges_df["weight"].max():.4f}',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig(artifact_dir / "graph_visualization_detailed.png", dpi=200, bbox_inches='tight')
print(f"\nVisualization saved to {artifact_dir / 'graph_visualization_detailed.png'}")

# Additional analysis: Find hub nodes
print("\n=== Hub Nodes (highest degree) ===")
degree_dict = dict(graph.degree())
top_hubs = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
for node, degree in top_hubs:
    print(f"  Node {node}: degree={degree}")

print("\n=== Most attended edges ===")
for _, row in edges_df_sorted.head(15).iterrows():
    print(f"  {int(row['source'])} -> {int(row['target'])}: weight={row['weight']:.4f}")

plt.show()
