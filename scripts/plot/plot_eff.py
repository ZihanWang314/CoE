import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
from matplotlib.ticker import ScalarFormatter

# Set better aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.edgecolor'] = '#333333'

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1.1, 1]})

# ================ LEFT SUBPLOT - CONVERGENCE PERFORMANCE ================
# Load the data
data = pd.read_csv('data3.tsv', sep='\t')

# Extract steps and values
steps = data['Step'].values
# For the left plot, we'll focus on the models mentioned in your requirements
# CoE-2(4/48) and MoE(8/64)
# Assuming these are in your columns, adjust as needed
left_columns = ['64ept-8tpk-1itr', '64ept-4tpk-2itr', '48ept-4tpk-2itr']
left_labels = ['MoE (K=8, N=64)', 'CoE (C=2, K=4, N=64)', 'CoE (C=2, K=4, N=48)']

# Find the index where steps >= 100
start_idx = np.where(steps >= 100)[0]
start_idx = 0 if len(start_idx) == 0 else start_idx[0]

# Define colors and markers for left plot
left_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
left_markers = ['o', 's', 'D']
left_linestyles = ['-', '--', ':']

# Create plot with log scales for selected columns
for i, (column, label) in enumerate(zip(left_columns, left_labels)):
    if column in data.columns:
        ax1.loglog(steps[start_idx:], data[column].values[start_idx:], 
                  label=label, 
                  linestyle=left_linestyles[i], 
                  linewidth=2.5,
                  marker=left_markers[i],
                  markersize=5,
                  markevery=5,
                  color=left_colors[i])

# Add labels and title for left plot
ax1.set_xlabel('Steps', fontweight='bold', fontsize=14)
ax1.set_ylabel('Validation Loss', fontweight='bold', fontsize=14)
ax1.set_title('Performance Comparison', fontweight='bold', fontsize=16)
ax1.legend(frameon=True, fontsize=12, framealpha=0.7, edgecolor='#333333', loc='upper right')
ax1.grid(True, which='minor', linestyle=':', alpha=0.4)
ax1.grid(True, which='major', linestyle='-', alpha=0.5)

ax1.set_xticks([100, 200, 300, 400, 600, 800, 1000])
ax1.set_yticks([1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3])
ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax1.xaxis.get_major_formatter().set_scientific(False)
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax1.yaxis.get_major_formatter().set_scientific(False)

# Adjust axis limits to focus on the important part
ax1.set_xlim(190, 1000)
ax1.set_ylim(1, 3)


ax1.set_facecolor('#f8f9fa')

# ================ RIGHT SUBPLOT - EFFICIENCY METRICS ================
# Create data for the right plot - Resource Efficiency
models = ['CoE (C=2, K=4, N=48)', 'MoE (K=8, N=64)', 'CoE (C=2, K=4, N=64)']
params = [412.63, 544.51, 544.75]  # in million parameters
memory = [9.64, 11.70, 11.70]  # in GB

# Normalize the values relative to MoE(8/64)
params_norm = [p/params[1]*100 for p in params]
memory_norm = [m/memory[1]*100 for m in memory]

# Set width of bars
barWidth = 0.25

# Set positions of the bars on X axis
r1 = np.arange(len(models))
r2 = [x + barWidth for x in r1]

# Create grouped bars
ax2.bar(r1, params_norm, width=barWidth, label='Parameters', color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1)
ax2.bar(r2, memory_norm, width=barWidth, label='Memory', color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1)

# Show the reference line at 100%
ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, linewidth=1)

# Add values on the bars
for i, (p, m) in enumerate(zip(params_norm, memory_norm)):
    p, m = round(p, 1), round(m, 1)
    if p != 100.0:
        ax2.text(r1[i], p+1, f"{p:.1f}%", ha='center', va='bottom', fontsize=10)
    else:
        ax2.text(r1[i], p+1, f"{p:.0f}%", ha='center', va='bottom', fontsize=10)
    if m != 100.0:
        ax2.text(r2[i], m+1, f"{m:.1f}%", ha='center', va='bottom', fontsize=10)
    else:
        ax2.text(r2[i], m+1, f"{m:.0f}%", ha='center', va='bottom', fontsize=10)

# Highlight the memory reduction of CoE-2(4/48)
memory_reduction = 100 - memory_norm[0]
ax2.annotate(f"-{memory_reduction:.1f}% memory", 
            xy=(r2[0], memory_norm[0]+20), 
            xytext=(r2[0]-0.3, memory_norm[0]+10),
            fontsize=11,
            weight='bold',
            color='#2ca02c',
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.2, fc="#2ca02c", ec="none"))

# Add xticks for each model group
ax2.set_xticks([r + barWidth/2 for r in range(len(models))])
ax2.set_xticklabels(models)
# make it smaller
ax2.tick_params(axis='x', labelsize=10)

# Set labels and title for right plot
ax2.set_ylabel('Relative to MoE(8/64) (%)', fontweight='bold', fontsize=14)
ax2.set_title('Resource Efficiency', fontweight='bold', fontsize=16)
ax2.legend(frameon=True, fontsize=12, framealpha=0.7, edgecolor='#333333', loc='upper right')
ax2.set_ylim(0, 140)
ax2.set_facecolor('#f8f9fa')

# Add gridlines
ax2.grid(axis='y', linestyle='-', alpha=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Main title for the entire figure
# plt.suptitle('CoE Reduces Memory Requirements while Maintaining Performance', 
            # fontweight='bold', fontsize=18, y=0.98)

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('plot_eff.png', dpi=300, bbox_inches='tight')
plt.show()