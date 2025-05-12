import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

plt.style.use('seaborn-v0_8-whitegrid')
# mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.edgecolor'] = '#333333'

# Data extraction from the provided table
steps = np.array([i for i in range(10, 1010, 10)])

# Read data from the TSV file
import pandas as pd

# Load the data
data = pd.read_csv('data.tsv', sep='\t')

# Extract steps and values
steps = data['Step'].values
moe_values = data['64ept-8tpk-1itr'].values  # MoE data (assuming this is MoE)
coe_values = data['64ept-8tpk-2itr'].values  # CoE data (assuming this is CoE-2)


# Plot
plt.figure(figsize=(12*0.5, 7*0.5))

# Only include data from step 100 onward
start_idx = 4  # Index for step 100

# Create plot with log scales
plt.loglog(steps[start_idx:], moe_values[start_idx:], 
           label='MoE (K=8, C=1)',
           linestyle='-', 
           linewidth=2.5,
           marker='o',
           markersize=5,
           markevery=5,
           color='#1f77b4')

plt.loglog(steps[start_idx:], coe_values[start_idx:], 
           label='CoE (K=4, C=2)', 
           linestyle='--', 
           linewidth=2.5,
           marker='s',
           markersize=5,
           markevery=5,
           color='#ff7f0e')

# Add better labels and title
plt.xlabel('Steps', fontweight='bold', fontsize=14)
plt.ylabel('Validation Loss', fontweight='bold', fontsize=14)
# plt.title('Convergence: MoE vs CoE', 
        #   fontweight='bold', fontsize=16, pad=10)
plt.gca().set_xticks([100, 200, 400, 600, 800, 1000])
plt.gca().set_yticks([1, 2, 3, 4])
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_major_formatter(ScalarFormatter())

# Improve legend
plt.legend(frameon=True, fontsize=12, framealpha=0.7, 
           edgecolor='#333333', loc='upper right')

# Add minor grid lines
plt.grid(True, which='minor', linestyle=':', alpha=0.4)
plt.grid(True, which='major', linestyle='-', alpha=0.5)

# Add annotations for key points
plt.annotate('Faster initial\nconvergence', 
             xy=(150, 2.78), 
             xytext=(170, 3.2),
             arrowprops=dict(arrowstyle='->'),
             fontsize=12)

# Adjust axis limits to focus on the important part
plt.xlim(90, 1000)
plt.ylim(1, 4)

# Add a subtle background color
plt.gca().set_facecolor('#f8f9fa')

# Save with higher DPI for better quality
plt.tight_layout()
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
# plt.show()