"""
Gym Network Simulation: The Role of Weak Ties in Sustaining Engagement
======================================================================
INFOI606 - Network Science Course Project
Author: Devan
Description: Simulates a gym membership network to test whether weak ties
(casual cross-group interactions) play a more significant role than strong
ties (close workout partnerships) in sustaining engagement and retention.

Compares simple contagion (SI model) vs complex contagion (threshold model)
to test whether weak ties operate differently in behavioral engagement
contexts vs information-seeking contexts.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# 1. NETWORK CONSTRUCTION
# ============================================================

def build_gym_network(n_clusters=5, members_per_cluster=35, 
                       intra_p=0.4, weak_tie_count=25, seed=SEED):
    """
    Build a simulated gym membership network.
    
    Parameters:
    -----------
    n_clusters : int
        Number of social clusters (e.g., morning regulars, powerlifters, 
        yoga class, evening cardio, CrossFit group)
    members_per_cluster : int
        Members per cluster
    intra_p : float
        Probability of edge within a cluster (strong ties)
    weak_tie_count : int
        Number of random cross-cluster edges (weak ties)
    
    Returns:
    --------
    G : networkx.Graph
        The full network with weak ties
    G_strong : networkx.Graph
        The network with only strong ties (no bridges)
    cluster_labels : dict
        Node -> cluster mapping
    weak_edges : list
        List of weak tie edges
    """
    G = nx.Graph()
    cluster_labels = {}
    cluster_names = ['Morning Regulars', 'Powerlifters', 'Yoga Class', 
                     'Evening Cardio', 'CrossFit Group']
    
    # Phase 1: Build clustered base network (strong ties)
    node_id = 0
    clusters = []
    for c in range(n_clusters):
        cluster_nodes = []
        for _ in range(members_per_cluster):
            G.add_node(node_id, cluster=c, cluster_name=cluster_names[c])
            cluster_labels[node_id] = c
            cluster_nodes.append(node_id)
            node_id += 1
        clusters.append(cluster_nodes)
        
        # Add intra-cluster edges (strong ties)
        for i, n1 in enumerate(cluster_nodes):
            for n2 in cluster_nodes[i+1:]:
                if random.random() < intra_p:
                    G.add_edge(n1, n2, weight=random.uniform(0.6, 1.0), 
                              tie_type='strong')
    
    # Phase 2: Add weak ties (cross-cluster bridges)
    weak_edges = []
    attempts = 0
    while len(weak_edges) < weak_tie_count and attempts < weak_tie_count * 10:
        c1, c2 = random.sample(range(n_clusters), 2)
        n1 = random.choice(clusters[c1])
        n2 = random.choice(clusters[c2])
        if not G.has_edge(n1, n2):
            G.add_edge(n1, n2, weight=random.uniform(0.1, 0.4), 
                      tie_type='weak')
            weak_edges.append((n1, n2))
        attempts += 1
    
    # Create strong-ties-only version
    G_strong = G.copy()
    G_strong.remove_edges_from(weak_edges)
    
    print(f"Network built:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Total edges: {G.number_of_edges()}")
    print(f"  Strong ties: {G.number_of_edges() - len(weak_edges)}")
    print(f"  Weak ties: {len(weak_edges)}")
    print(f"  Clusters: {n_clusters} x {members_per_cluster} members")
    
    return G, G_strong, cluster_labels, weak_edges


# ============================================================
# 2. NETWORK METRICS
# ============================================================

def compute_metrics(G, label="Network"):
    """Compute key network metrics."""
    metrics = {}
    
    # Basic stats
    metrics['nodes'] = G.number_of_nodes()
    metrics['edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    # Clustering coefficient
    metrics['avg_clustering'] = nx.average_clustering(G)
    
    # Connected components
    components = list(nx.connected_components(G))
    metrics['n_components'] = len(components)
    metrics['largest_component'] = len(max(components, key=len))
    metrics['largest_component_pct'] = metrics['largest_component'] / metrics['nodes'] * 100
    
    # Average path length (within largest component)
    largest_cc = G.subgraph(max(components, key=len)).copy()
    if largest_cc.number_of_nodes() > 1:
        metrics['avg_path_length'] = nx.average_shortest_path_length(largest_cc)
    else:
        metrics['avg_path_length'] = float('inf')
    
    # Average degree
    degrees = [d for n, d in G.degree()]
    metrics['avg_degree'] = np.mean(degrees)
    
    # Betweenness centrality (top 10 nodes)
    bc = nx.betweenness_centrality(G)
    metrics['max_betweenness'] = max(bc.values())
    metrics['avg_betweenness'] = np.mean(list(bc.values()))
    
    print(f"\n--- {label} ---")
    print(f"  Nodes: {metrics['nodes']}, Edges: {metrics['edges']}")
    print(f"  Density: {metrics['density']:.4f}")
    print(f"  Avg clustering coefficient: {metrics['avg_clustering']:.4f}")
    print(f"  Connected components: {metrics['n_components']}")
    print(f"  Largest component: {metrics['largest_component']} ({metrics['largest_component_pct']:.1f}%)")
    print(f"  Avg path length (largest CC): {metrics['avg_path_length']:.3f}")
    print(f"  Avg degree: {metrics['avg_degree']:.2f}")
    print(f"  Avg betweenness centrality: {metrics['avg_betweenness']:.5f}")
    
    return metrics


# ============================================================
# 3. DIFFUSION MODELS
# ============================================================

def simple_contagion_SI(G, seed_node, beta=0.15, max_steps=50):
    """
    Simple contagion (SI model).
    A node becomes engaged after a single contact with an engaged neighbor.
    
    Parameters:
    -----------
    G : networkx.Graph
    seed_node : int - initial engaged node
    beta : float - transmission probability per edge per time step
    max_steps : int - maximum simulation steps
    
    Returns:
    --------
    history : list of sets - engaged nodes at each time step
    """
    engaged = {seed_node}
    history = [engaged.copy()]
    
    for t in range(max_steps):
        new_engaged = set()
        for node in engaged:
            for neighbor in G.neighbors(node):
                if neighbor not in engaged and neighbor not in new_engaged:
                    if random.random() < beta:
                        new_engaged.add(neighbor)
        
        if not new_engaged:
            # Pad remaining steps
            for _ in range(max_steps - t - 1):
                history.append(engaged.copy())
            break
        
        engaged = engaged | new_engaged
        history.append(engaged.copy())
    
    return history


def complex_contagion_threshold(G, seed_nodes, threshold=0.3, max_steps=50):
    """
    Complex contagion (threshold model).
    A node becomes engaged only when a fraction >= threshold of its 
    neighbors are already engaged.
    
    Parameters:
    -----------
    G : networkx.Graph
    seed_nodes : set - initial engaged nodes (need multiple seeds for complex contagion)
    threshold : float - fraction of neighbors required
    max_steps : int
    
    Returns:
    --------
    history : list of sets
    """
    engaged = set(seed_nodes)
    history = [engaged.copy()]
    
    for t in range(max_steps):
        new_engaged = set()
        for node in G.nodes():
            if node not in engaged:
                neighbors = set(G.neighbors(node))
                if len(neighbors) == 0:
                    continue
                engaged_neighbors = neighbors & engaged
                if len(engaged_neighbors) / len(neighbors) >= threshold:
                    new_engaged.add(node)
        
        if not new_engaged:
            for _ in range(max_steps - t - 1):
                history.append(engaged.copy())
            break
        
        engaged = engaged | new_engaged
        history.append(engaged.copy())
    
    return history


def run_diffusion_experiments(G, G_strong, cluster_labels, n_runs=20, max_steps=50):
    """
    Run diffusion experiments across both network conditions and both models.
    Averages results over multiple runs with different seed nodes.
    """
    results = {}
    n_nodes = G.number_of_nodes()
    nodes_list = list(G.nodes())
    
    for net_label, network in [("With Weak Ties", G), ("Strong Ties Only", G_strong)]:
        for model_label in ["Simple Contagion (SI)", "Complex Contagion (Threshold)"]:
            key = f"{net_label} | {model_label}"
            all_curves = []
            
            for run in range(n_runs):
                # Pick seed node(s) from a random cluster
                seed = random.choice(nodes_list)
                
                if "Simple" in model_label:
                    history = simple_contagion_SI(network, seed, beta=0.15, max_steps=max_steps)
                else:
                    # For complex contagion, seed a small group in one cluster
                    cluster = cluster_labels[seed]
                    same_cluster = [n for n in nodes_list if cluster_labels[n] == cluster]
                    seed_group = set(random.sample(same_cluster, min(5, len(same_cluster))))
                    history = complex_contagion_threshold(network, seed_group, 
                                                          threshold=0.3, max_steps=max_steps)
                
                # Convert to fraction curve
                curve = [len(h) / n_nodes for h in history]
                # Pad to max_steps if needed
                while len(curve) < max_steps + 1:
                    curve.append(curve[-1])
                all_curves.append(curve[:max_steps + 1])
            
            # Average across runs
            avg_curve = np.mean(all_curves, axis=0)
            std_curve = np.std(all_curves, axis=0)
            results[key] = {'avg': avg_curve, 'std': std_curve}
            
            # Report key stats
            final_reach = avg_curve[-1] * 100
            # Time to 50% (if reached)
            t50 = next((i for i, v in enumerate(avg_curve) if v >= 0.5), None)
            print(f"\n{key}:")
            print(f"  Final reach: {final_reach:.1f}%")
            print(f"  Time to 50%: {t50 if t50 else 'Not reached'}")
    
    return results


# ============================================================
# 4. PROGRESSIVE WEAK TIE REMOVAL
# ============================================================

def progressive_removal(G, weak_edges, cluster_labels, steps=6):
    """
    Progressively remove weak ties and measure impact on metrics.
    """
    removal_fractions = np.linspace(0, 1, steps)
    results = []
    
    print("\n=== Progressive Weak Tie Removal ===")
    for frac in removal_fractions:
        G_temp = G.copy()
        n_remove = int(frac * len(weak_edges))
        edges_to_remove = random.sample(weak_edges, n_remove) if n_remove > 0 else []
        G_temp.remove_edges_from(edges_to_remove)
        
        # Compute metrics
        components = list(nx.connected_components(G_temp))
        largest_cc = G_temp.subgraph(max(components, key=len)).copy()
        
        result = {
            'frac_removed': frac,
            'n_removed': n_remove,
            'n_components': len(components),
            'largest_cc_pct': len(max(components, key=len)) / G_temp.number_of_nodes() * 100,
            'avg_clustering': nx.average_clustering(G_temp),
        }
        
        if largest_cc.number_of_nodes() > 1:
            result['avg_path_length'] = nx.average_shortest_path_length(largest_cc)
        else:
            result['avg_path_length'] = float('inf')
        
        results.append(result)
        print(f"  Removed {frac*100:.0f}% weak ties ({n_remove}): "
              f"Components={result['n_components']}, "
              f"Largest CC={result['largest_cc_pct']:.1f}%, "
              f"Avg Path={result['avg_path_length']:.2f}")
    
    return results


# ============================================================
# 5. VISUALIZATIONS
# ============================================================

def plot_network_comparison(G, G_strong, cluster_labels, weak_edges):
    """Plot side-by-side network visualizations."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Color map for clusters
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    node_colors = [colors[cluster_labels[n]] for n in G.nodes()]
    
    # Layout (use same positions for both)
    pos = nx.spring_layout(G, k=0.3, seed=SEED, iterations=50)
    
    # With weak ties
    ax = axes[0]
    strong_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('tie_type') == 'strong']
    weak_edge_list = [(u, v) for u, v, d in G.edges(data=True) if d.get('tie_type') == 'weak']
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=30, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=strong_edges, ax=ax, alpha=0.15, 
                           edge_color='gray', width=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=weak_edge_list, ax=ax, alpha=0.8, 
                           edge_color='red', width=1.5, style='dashed')
    ax.set_title('With Weak Ties (red dashed = bridges)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Strong ties only
    ax = axes[1]
    nx.draw_networkx_nodes(G_strong, pos, ax=ax, node_size=30, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G_strong, pos, ax=ax, alpha=0.15, edge_color='gray', width=0.5)
    ax.set_title('Strong Ties Only (no bridges)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Legend
    patches = [mpatches.Patch(color=colors[i], label=name) 
               for i, name in enumerate(['Morning Regulars', 'Powerlifters', 
                                          'Yoga Class', 'Evening Cardio', 'CrossFit'])]
    fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=10)
    
    plt.suptitle('Gym Member Network: Effect of Weak Ties on Structure', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('fig1_network_comparison.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("Saved: fig1_network_comparison.png")


def plot_diffusion_curves(results, max_steps=50):
    """Plot diffusion curves for all conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors_map = {
        'With Weak Ties': '#2ecc71',
        'Strong Ties Only': '#e74c3c'
    }
    
    time = np.arange(max_steps + 1)
    
    for idx, model in enumerate(["Simple Contagion (SI)", "Complex Contagion (Threshold)"]):
        ax = axes[idx]
        for net_label in ["With Weak Ties", "Strong Ties Only"]:
            key = f"{net_label} | {model}"
            avg = results[key]['avg']
            std = results[key]['std']
            color = colors_map[net_label]
            
            ax.plot(time, avg * 100, label=net_label, color=color, linewidth=2)
            ax.fill_between(time, (avg - std) * 100, (avg + std) * 100, 
                          alpha=0.2, color=color)
        
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('% Network Engaged', fontsize=12)
        ax.set_title(model, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Engagement Diffusion: Simple vs Complex Contagion', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig2_diffusion_curves.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("Saved: fig2_diffusion_curves.png")


def plot_metrics_comparison(metrics_with, metrics_without):
    """Bar chart comparing key metrics."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    labels = ['With Weak Ties', 'Strong Ties Only']
    colors = ['#2ecc71', '#e74c3c']
    
    # Avg Path Length
    vals = [metrics_with['avg_path_length'], metrics_without['avg_path_length']]
    axes[0].bar(labels, vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_title('Avg Path Length', fontweight='bold')
    axes[0].set_ylabel('Steps')
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Clustering Coefficient
    vals = [metrics_with['avg_clustering'], metrics_without['avg_clustering']]
    axes[1].bar(labels, vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_title('Avg Clustering Coefficient', fontweight='bold')
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Largest Connected Component
    vals = [metrics_with['largest_component_pct'], metrics_without['largest_component_pct']]
    axes[2].bar(labels, vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[2].set_title('Largest Component (%)', fontweight='bold')
    axes[2].set_ylabel('% of Network')
    for i, v in enumerate(vals):
        axes[2].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Number of Components
    vals = [metrics_with['n_components'], metrics_without['n_components']]
    axes[3].bar(labels, vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[3].set_title('Connected Components', fontweight='bold')
    axes[3].set_ylabel('Count')
    for i, v in enumerate(vals):
        axes[3].text(i, v + 0.2, f'{v}', ha='center', fontweight='bold')
    
    plt.suptitle('Structural Impact of Weak Tie Removal', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig3_metrics_comparison.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("Saved: fig3_metrics_comparison.png")


def plot_progressive_removal(removal_results):
    """Plot metrics as weak ties are progressively removed."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    fracs = [r['frac_removed'] * 100 for r in removal_results]
    
    # Path length
    axes[0].plot(fracs, [r['avg_path_length'] for r in removal_results], 
                'o-', color='#e74c3c', linewidth=2, markersize=8)
    axes[0].set_xlabel('% Weak Ties Removed')
    axes[0].set_ylabel('Avg Path Length')
    axes[0].set_title('Path Length vs Weak Tie Removal', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Largest CC
    axes[1].plot(fracs, [r['largest_cc_pct'] for r in removal_results], 
                'o-', color='#3498db', linewidth=2, markersize=8)
    axes[1].set_xlabel('% Weak Ties Removed')
    axes[1].set_ylabel('Largest Component (%)')
    axes[1].set_title('Connectivity vs Weak Tie Removal', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Components
    axes[2].plot(fracs, [r['n_components'] for r in removal_results], 
                'o-', color='#9b59b6', linewidth=2, markersize=8)
    axes[2].set_xlabel('% Weak Ties Removed')
    axes[2].set_ylabel('Number of Components')
    axes[2].set_title('Fragmentation vs Weak Tie Removal', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Progressive Weak Tie Removal: Network Degradation', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig4_progressive_removal.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("Saved: fig4_progressive_removal.png")


def plot_degree_distribution(G, G_strong):
    """Plot degree distributions for both networks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, network, label, color in [(axes[0], G, 'With Weak Ties', '#2ecc71'),
                                       (axes[1], G_strong, 'Strong Ties Only', '#e74c3c')]:
        degrees = [d for n, d in network.degree()]
        ax.hist(degrees, bins=range(0, max(degrees)+2), color=color, 
                edgecolor='black', alpha=0.7, density=True)
        ax.set_xlabel('Degree', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{label}\n(mean={np.mean(degrees):.1f}, max={max(degrees)})', 
                     fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Degree Distribution Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig5_degree_distribution.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("Saved: fig5_degree_distribution.png")


# ============================================================
# 6. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GYM NETWORK SIMULATION")
    print("The Role of Weak Ties in Sustaining Engagement")
    print("=" * 60)
    
    # --- Build Network ---
    print("\n[1/6] Building gym network...")
    G, G_strong, cluster_labels, weak_edges = build_gym_network()
    
    # --- Compute Metrics ---
    print("\n[2/6] Computing network metrics...")
    metrics_with = compute_metrics(G, "Full Network (with weak ties)")
    metrics_without = compute_metrics(G_strong, "Strong Ties Only")
    
    # --- Run Diffusion Experiments ---
    print("\n[3/6] Running diffusion experiments (20 runs each)...")
    diffusion_results = run_diffusion_experiments(G, G_strong, cluster_labels)
    
    # --- Progressive Removal ---
    print("\n[4/6] Progressive weak tie removal analysis...")
    removal_results = progressive_removal(G, weak_edges)
    
    # --- Generate Visualizations ---
    print("\n[5/6] Generating visualizations...")
    plot_network_comparison(G, G_strong, cluster_labels, weak_edges)
    plot_diffusion_curves(diffusion_results)
    plot_metrics_comparison(metrics_with, metrics_without)
    plot_progressive_removal(removal_results)
    plot_degree_distribution(G, G_strong)
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("[6/6] SUMMARY OF FINDINGS")
    print("=" * 60)
    
    path_diff = metrics_without['avg_path_length'] - metrics_with['avg_path_length']
    print(f"\nPath length increase without weak ties: +{path_diff:.2f} steps")
    print(f"Largest component WITH weak ties: {metrics_with['largest_component_pct']:.1f}%")
    print(f"Largest component WITHOUT weak ties: {metrics_without['largest_component_pct']:.1f}%")
    print(f"Components WITH: {metrics_with['n_components']} | WITHOUT: {metrics_without['n_components']}")
    
    print("\nDiffusion final reach:")
    for key, data in diffusion_results.items():
        print(f"  {key}: {data['avg'][-1]*100:.1f}%")
    
    print("\nFigures saved:")
    print("  fig1_network_comparison.png")
    print("  fig2_diffusion_curves.png")
    print("  fig3_metrics_comparison.png")
    print("  fig4_progressive_removal.png")
    print("  fig5_degree_distribution.png")
    print("\nDone!")
