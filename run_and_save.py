"""
Run the gym network simulation and save all results to a text file for the report.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import json

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

results_log = []
def log(msg):
    print(msg)
    results_log.append(msg)

# ============================================================
# 1. BUILD NETWORK
# ============================================================
log("=" * 60)
log("BUILDING GYM NETWORK")
log("=" * 60)

n_clusters = 5
members_per_cluster = 35
intra_p = 0.4
weak_tie_count = 25

G = nx.Graph()
cluster_labels = {}
cluster_names = ['Morning Regulars', 'Powerlifters', 'Yoga Class', 'Evening Cardio', 'CrossFit Group']

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
    for i, n1 in enumerate(cluster_nodes):
        for n2 in cluster_nodes[i+1:]:
            if random.random() < intra_p:
                G.add_edge(n1, n2, weight=random.uniform(0.6, 1.0), tie_type='strong')

weak_edges = []
attempts = 0
while len(weak_edges) < weak_tie_count and attempts < weak_tie_count * 10:
    c1, c2 = random.sample(range(n_clusters), 2)
    n1 = random.choice(clusters[c1])
    n2 = random.choice(clusters[c2])
    if not G.has_edge(n1, n2):
        G.add_edge(n1, n2, weight=random.uniform(0.1, 0.4), tie_type='weak')
        weak_edges.append((n1, n2))
    attempts += 1

G_strong = G.copy()
G_strong.remove_edges_from(weak_edges)

log(f"Nodes: {G.number_of_nodes()}")
log(f"Total edges: {G.number_of_edges()}")
log(f"Strong ties: {G.number_of_edges() - len(weak_edges)}")
log(f"Weak ties: {len(weak_edges)}")

# ============================================================
# 2. COMPUTE METRICS
# ============================================================
log("\n" + "=" * 60)
log("NETWORK METRICS")
log("=" * 60)

def compute_metrics(net, label):
    m = {}
    m['density'] = nx.density(net)
    m['avg_clustering'] = nx.average_clustering(net)
    components = list(nx.connected_components(net))
    m['n_components'] = len(components)
    m['largest_component'] = len(max(components, key=len))
    m['largest_component_pct'] = m['largest_component'] / net.number_of_nodes() * 100
    largest_cc = net.subgraph(max(components, key=len)).copy()
    m['avg_path_length'] = nx.average_shortest_path_length(largest_cc) if largest_cc.number_of_nodes() > 1 else float('inf')
    degrees = [d for n, d in net.degree()]
    m['avg_degree'] = np.mean(degrees)
    bc = nx.betweenness_centrality(net)
    m['avg_betweenness'] = np.mean(list(bc.values()))
    
    log(f"\n--- {label} ---")
    log(f"  Density: {m['density']:.4f}")
    log(f"  Avg clustering: {m['avg_clustering']:.4f}")
    log(f"  Components: {m['n_components']}")
    log(f"  Largest component: {m['largest_component']} ({m['largest_component_pct']:.1f}%)")
    log(f"  Avg path length: {m['avg_path_length']:.3f}")
    log(f"  Avg degree: {m['avg_degree']:.2f}")
    log(f"  Avg betweenness: {m['avg_betweenness']:.5f}")
    return m

metrics_with = compute_metrics(G, "With Weak Ties")
metrics_without = compute_metrics(G_strong, "Strong Ties Only")

# ============================================================
# 3. DIFFUSION
# ============================================================
log("\n" + "=" * 60)
log("DIFFUSION EXPERIMENTS")
log("=" * 60)

def simple_contagion(net, seed_node, beta=0.15, max_steps=50):
    engaged = {seed_node}
    history = [engaged.copy()]
    for t in range(max_steps):
        new = set()
        for node in engaged:
            for nb in net.neighbors(node):
                if nb not in engaged and nb not in new and random.random() < beta:
                    new.add(nb)
        if not new:
            for _ in range(max_steps - t - 1): history.append(engaged.copy())
            break
        engaged = engaged | new
        history.append(engaged.copy())
    return history

def complex_contagion(net, seeds, threshold=0.3, max_steps=50):
    engaged = set(seeds)
    history = [engaged.copy()]
    for t in range(max_steps):
        new = set()
        for node in net.nodes():
            if node not in engaged:
                nbs = set(net.neighbors(node))
                if len(nbs) > 0 and len(nbs & engaged) / len(nbs) >= threshold:
                    new.add(node)
        if not new:
            for _ in range(max_steps - t - 1): history.append(engaged.copy())
            break
        engaged = engaged | new
        history.append(engaged.copy())
    return history

n_runs = 20
max_steps = 50
n_nodes = G.number_of_nodes()
nodes_list = list(G.nodes())
diffusion_results = {}

for net_label, network in [("With Weak Ties", G), ("Strong Ties Only", G_strong)]:
    for model_label in ["Simple Contagion (SI)", "Complex Contagion (Threshold)"]:
        key = f"{net_label} | {model_label}"
        all_curves = []
        for run in range(n_runs):
            seed = random.choice(nodes_list)
            if "Simple" in model_label:
                history = simple_contagion(network, seed)
            else:
                cluster = cluster_labels[seed]
                same = [n for n in nodes_list if cluster_labels[n] == cluster]
                seed_group = set(random.sample(same, min(5, len(same))))
                history = complex_contagion(network, seed_group)
            curve = [len(h) / n_nodes for h in history]
            while len(curve) < max_steps + 1: curve.append(curve[-1])
            all_curves.append(curve[:max_steps + 1])
        avg = np.mean(all_curves, axis=0)
        std = np.std(all_curves, axis=0)
        diffusion_results[key] = {'avg': avg, 'std': std}
        t50 = next((i for i, v in enumerate(avg) if v >= 0.5), None)
        log(f"\n{key}:")
        log(f"  Final reach: {avg[-1]*100:.1f}%")
        log(f"  Time to 50%: {t50 if t50 else 'Not reached'}")

# ============================================================
# 4. PROGRESSIVE REMOVAL
# ============================================================
log("\n" + "=" * 60)
log("PROGRESSIVE WEAK TIE REMOVAL")
log("=" * 60)

removal_results = []
for frac in np.linspace(0, 1, 6):
    G_temp = G.copy()
    n_remove = int(frac * len(weak_edges))
    edges_remove = random.sample(weak_edges, n_remove) if n_remove > 0 else []
    G_temp.remove_edges_from(edges_remove)
    comps = list(nx.connected_components(G_temp))
    largest_cc = G_temp.subgraph(max(comps, key=len)).copy()
    apl = nx.average_shortest_path_length(largest_cc) if largest_cc.number_of_nodes() > 1 else float('inf')
    r = {'frac': frac, 'n_remove': n_remove, 'n_components': len(comps),
         'largest_cc_pct': len(max(comps, key=len)) / G_temp.number_of_nodes() * 100,
         'avg_path_length': apl}
    removal_results.append(r)
    log(f"  Removed {frac*100:.0f}%: Components={r['n_components']}, LCC={r['largest_cc_pct']:.1f}%, Path={apl:.2f}")

# ============================================================
# 5. FIGURES
# ============================================================
log("\n" + "=" * 60)
log("GENERATING FIGURES")
log("=" * 60)

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
node_colors = [colors[cluster_labels[n]] for n in G.nodes()]
pos = nx.spring_layout(G, k=0.3, seed=SEED, iterations=50)

# Fig 1: Network comparison
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
strong_e = [(u,v) for u,v,d in G.edges(data=True) if d.get('tie_type')=='strong']
weak_e = [(u,v) for u,v,d in G.edges(data=True) if d.get('tie_type')=='weak']
nx.draw_networkx_nodes(G, pos, ax=axes[0], node_size=30, node_color=node_colors, alpha=0.8)
nx.draw_networkx_edges(G, pos, edgelist=strong_e, ax=axes[0], alpha=0.15, edge_color='gray', width=0.5)
nx.draw_networkx_edges(G, pos, edgelist=weak_e, ax=axes[0], alpha=0.8, edge_color='red', width=1.5, style='dashed')
axes[0].set_title('With Weak Ties (red dashed)', fontsize=14, fontweight='bold')
axes[0].axis('off')
nx.draw_networkx_nodes(G_strong, pos, ax=axes[1], node_size=30, node_color=node_colors, alpha=0.8)
nx.draw_networkx_edges(G_strong, pos, ax=axes[1], alpha=0.15, edge_color='gray', width=0.5)
axes[1].set_title('Strong Ties Only', fontsize=14, fontweight='bold')
axes[1].axis('off')
patches = [mpatches.Patch(color=colors[i], label=n) for i,n in enumerate(cluster_names)]
fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=10)
plt.tight_layout()
plt.savefig('fig1_network_comparison.png', dpi=200, bbox_inches='tight')
log("Saved fig1")

# Fig 2: Diffusion curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
time = np.arange(max_steps + 1)
cm = {'With Weak Ties': '#2ecc71', 'Strong Ties Only': '#e74c3c'}
for idx, model in enumerate(["Simple Contagion (SI)", "Complex Contagion (Threshold)"]):
    ax = axes[idx]
    for nl in ["With Weak Ties", "Strong Ties Only"]:
        k = f"{nl} | {model}"
        avg = diffusion_results[k]['avg']
        std = diffusion_results[k]['std']
        ax.plot(time, avg*100, label=nl, color=cm[nl], linewidth=2)
        ax.fill_between(time, (avg-std)*100, (avg+std)*100, alpha=0.2, color=cm[nl])
    ax.set_xlabel('Time Steps'); ax.set_ylabel('% Engaged')
    ax.set_title(model, fontsize=14, fontweight='bold')
    ax.legend(); ax.set_ylim(0, 105); ax.grid(True, alpha=0.3)
plt.suptitle('Engagement Diffusion', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('fig2_diffusion_curves.png', dpi=200, bbox_inches='tight')
log("Saved fig2")

# Fig 3: Metrics comparison
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
labels = ['With Weak Ties', 'Strong Ties Only']
cols = ['#2ecc71', '#e74c3c']
for ax, key, ylabel in [(axes[0], 'avg_path_length', 'Steps'), (axes[1], 'avg_clustering', ''),
                         (axes[2], 'largest_component_pct', '% of Network'), (axes[3], 'n_components', 'Count')]:
    vals = [metrics_with[key], metrics_without[key]]
    ax.bar(labels, vals, color=cols, edgecolor='black', linewidth=0.5)
    ax.set_title(key.replace('_', ' ').title(), fontweight='bold')
    if ylabel: ax.set_ylabel(ylabel)
    for i, v in enumerate(vals):
        fmt = f'{v:.2f}' if isinstance(v, float) else str(v)
        ax.text(i, v * 1.02, fmt, ha='center', fontweight='bold')
plt.suptitle('Structural Impact of Weak Ties', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('fig3_metrics_comparison.png', dpi=200, bbox_inches='tight')
log("Saved fig3")

# Fig 4: Progressive removal
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fracs = [r['frac']*100 for r in removal_results]
axes[0].plot(fracs, [r['avg_path_length'] for r in removal_results], 'o-', color='#e74c3c', linewidth=2)
axes[0].set_xlabel('% Weak Ties Removed'); axes[0].set_ylabel('Avg Path Length')
axes[0].set_title('Path Length', fontweight='bold'); axes[0].grid(True, alpha=0.3)
axes[1].plot(fracs, [r['largest_cc_pct'] for r in removal_results], 'o-', color='#3498db', linewidth=2)
axes[1].set_xlabel('% Weak Ties Removed'); axes[1].set_ylabel('Largest Component (%)')
axes[1].set_title('Connectivity', fontweight='bold'); axes[1].grid(True, alpha=0.3)
axes[2].plot(fracs, [r['n_components'] for r in removal_results], 'o-', color='#9b59b6', linewidth=2)
axes[2].set_xlabel('% Weak Ties Removed'); axes[2].set_ylabel('Components')
axes[2].set_title('Fragmentation', fontweight='bold'); axes[2].grid(True, alpha=0.3)
plt.suptitle('Progressive Weak Tie Removal', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('fig4_progressive_removal.png', dpi=200, bbox_inches='tight')
log("Saved fig4")

# Fig 5: Degree distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, net, lbl, col in [(axes[0], G, 'With Weak Ties', '#2ecc71'), (axes[1], G_strong, 'Strong Ties Only', '#e74c3c')]:
    degs = [d for n,d in net.degree()]
    ax.hist(degs, bins=range(0, max(degs)+2), color=col, edgecolor='black', alpha=0.7, density=True)
    ax.set_xlabel('Degree'); ax.set_ylabel('Frequency')
    ax.set_title(f'{lbl} (mean={np.mean(degs):.1f})', fontweight='bold'); ax.grid(True, alpha=0.3)
plt.suptitle('Degree Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('fig5_degree_distribution.png', dpi=200, bbox_inches='tight')
log("Saved fig5")

# Save results log
with open('simulation_results.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(results_log))
log("\nDONE - All results saved to simulation_results.txt")
