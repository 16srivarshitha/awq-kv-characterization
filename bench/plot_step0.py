import os
import matplotlib.pyplot as plt
import numpy as np

# ensure directory exists
os.makedirs("results/plots", exist_ok=True)

# exact data from your step 0b run
labels = ['Padded\n(G=16)', 'SIMT FP16\n(G=3)', 'SIMT INT8\n(G=3)']
graph_latency = np.array([429.81, 150.39, 166.73])
eager_latency = np.array([493.42, 151.67, 167.52])
launch_tax = eager_latency - graph_latency

x = np.arange(len(labels))
width = 0.5

fig, ax = plt.subplots(figsize=(8, 6))

# plot hardware execution time (bottom)
p1 = ax.bar(x, graph_latency, width, label='Silicon Execution (CUDA Graph)', color='#1f77b4')
# plot cpu launch overhead (stacked on top)
p2 = ax.bar(x, launch_tax, width, bottom=graph_latency, label='CPU Launch Tax (Eager Overhead)', color='#ff7f0e')

ax.set_ylabel('Latency (µs)')
ax.set_title('Decode Attention: Eager vs. CUDA Graph Execution (RTX 2080 Ti)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# add text annotations
for i in range(len(labels)):
    ax.text(x[i], graph_latency[i]/2, f"{graph_latency[i]:.1f} µs", ha='center', va='center', color='white', fontweight='bold')
    if launch_tax[i] > 2.0:
        ax.text(x[i], graph_latency[i] + launch_tax[i]/2, f"+{launch_tax[i]:.1f} µs", ha='center', va='center', color='black')

plt.tight_layout()
plt.savefig("results/plots/step0_cudagraph.png", dpi=300)
print("plot successfully saved to results/plots/step0_cudagraph.png")
