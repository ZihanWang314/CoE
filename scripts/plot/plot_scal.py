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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={'width_ratios': [1, 1.2]})

# ================ LEFT SUBPLOT - CONVERGENCE PERFORMANCE ================
steps = np.array([i*10 for i in range(1, 101)])  # 10 to 1000 by 10s
data = pd.read_csv('data.tsv', sep='\t')

# Define column names and pretty labels
columns = ['64ept-8tpk-1itr', '64ept-16tpk-1itr', '64ept-24tpk-1itr', '64ept-8tpk-2itr']
left_labels = ['MoE(T=64, K=8)', 'MoE(T=64, K=16)', 'MoE(T=64, K=24)', 'CoE (T=64, C=2)']


# 修改为解释清楚的 legend 名
left_data = [data[col] for col in columns]

start_idx = np.where(steps >= 100)[0]
start_idx = 0 if len(start_idx) == 0 else start_idx[0]

left_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
left_markers = ['o', '^', 'D', 's']
left_linestyles = ['-', '-.', ':', '--']

# 改为线性坐标轴
for i, (data_series, label) in enumerate(zip(left_data, left_labels)):
    ax1.loglog(steps[start_idx:], data_series[start_idx:], 
             label=label, 
             linestyle=left_linestyles[i], 
             linewidth=2.5,
             marker=left_markers[i],
             markersize=5,
             markevery=5,
             color=left_colors[i])

# 坐标轴标题改为非 log 描述
ax1.set_xlabel('Steps', fontweight='bold', fontsize=14)
ax1.set_ylabel('Validation Loss', fontweight='bold', fontsize=14)

# Legend 修改后显示更新
ax1.legend(frameon=True, fontsize=10, framealpha=0.7, edgecolor='#333333', loc='upper right')
ax1.grid(True, which='minor', linestyle=':', alpha=0.4)
ax1.grid(True, which='major', linestyle='-', alpha=0.5)
ax1.set_xlim(370, 1000)
ax1.set_ylim(1, 2.5)
ax1.set_facecolor('#f8f9fa')

ax1.set_title('Performance Comparison', fontweight='bold', fontsize=16)
# 修改 tick 设置和格式
ax1.set_xticks([200, 300, 400, 600, 800, 1000])
ax1.set_yticks([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4])
ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax1.xaxis.get_major_formatter().set_scientific(False)
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax1.yaxis.get_major_formatter().set_scientific(False)


# ================ RIGHT SUBPLOT - EFFICIENCY METRICS ================
models = ['MoE(T=64, K=8)', 'CoE (T=64, C=2)', 'MoE(T=64, K=16)', 'MoE(T=64, K=24)']


params = [544.75, 544.51, 544.51, 544.51]
memory = [11.65, 11.70, 11.70, 11.70]
time = [989, 1755, 1451, 1852]

params_norm = [p/params[0]*100 for p in params]
memory_norm = [m/memory[0]*100 for m in memory]
time_norm = [t/time[0]*100 for t in time]

barWidth = 0.26
x = np.arange(len(models))

# Colors
bar_colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
resource_colors = ['#1f77b4', '#2ca02c', '#d62728']  # param / mem / time

# Bars
ax2.bar(x - barWidth, params_norm, width=barWidth, label='Parameters', color=resource_colors[0], edgecolor='black')
ax2.bar(x, memory_norm, width=barWidth, label='Memory', color=resource_colors[1], edgecolor='black')
ax2.bar(x + barWidth, time_norm, width=barWidth, label='Time', color=resource_colors[2], edgecolor='black')

# Bar labels
for i in range(len(models)):
    ax2.text(x[i] - barWidth, params_norm[i] + 5, f'{params_norm[i]:.0f}%', ha='center', fontsize=9)
    ax2.text(x[i], memory_norm[i] + 5, f'{memory_norm[i]:.0f}%', ha='center', fontsize=9)
    ax2.text(x[i] + barWidth, time_norm[i] + 5, f'{time_norm[i]:.0f}%', ha='center', fontsize=9)

# Axes and labels
ax2.set_ylabel('Relative Efficiency (%)', fontweight='bold', fontsize=13)
ax2.set_title('Resource Efficiency', fontweight='bold', fontsize=15)
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=0, fontsize=10)
ax2.set_ylim(0, 200)
ax2.axhline(y=100, color='gray', linestyle='--', linewidth=1)

# Legend
ax2.legend(loc='upper left', fontsize=10, frameon=True)
ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.set_facecolor('#f8f9fa')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)


# Main title for the entire figure
# plt.suptitle('Compute Scaling: #Iteration (CoE) > #MoE Layers', 
            # fontweight='bold', fontsize=18, y=0.98)

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('plot_scal.png', dpi=300, bbox_inches='tight')
plt.show()