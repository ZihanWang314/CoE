import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
from matplotlib.ticker import ScalarFormatter

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.edgecolor'] = '#333333'

fig, ax = plt.subplots(figsize=(12*0.5, 7*0.5))
data = pd.read_csv('data6.tsv', sep='\t')

data.columns = [col.strip() for col in data.columns]

steps = data['Step'].values

methods = [
    'dsv2_coe_3-ds_8ept-8tpk-2itr - val/loss',    # dense2
    'dsv2_coe_3-ds_8ept-8tpk-1itr - val/loss',    # dense1
    'dsv2_coe_3-cc_64ept-8tpk-2itr - val/loss',   # sparse2
    'dsv2_coe_3-fl_64ept-8tpk-1itr - val/loss'    # sparse1
]

labels = [
    'Dense (Total=K=8, C=2)',
    'Dense (Total=K=8, C=1)',
    'Sparse CoE (K=8, C=2)',
    'Sparse MoE (K=8, C=1)'
]

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
markers = ['o', 's', 'D', '^']
linestyles = ['-', '--', ':', '-.']

for i, (method, label) in enumerate(zip(methods, labels)):
    if method in data.columns:
        ax.loglog(steps, data[method].values, 
                 label=label, 
                 linestyle=linestyles[i], 
                 linewidth=2.5,
                 marker=markers[i],
                 markersize=6,
                 markevery=10,
                 color=colors[i])

ax.set_xlabel('Steps', fontweight='bold', fontsize=14)
ax.set_ylabel('Validation Loss', fontweight='bold', fontsize=14)
# ax.set_title('Sparse Models Benefit from Recurrent Processing', fontweight='bold', fontsize=16, pad=10)
ax.legend(frameon=True, fontsize=12, framealpha=0.7, edgecolor='#333333', loc='upper right')
ax.grid(True, which='minor', linestyle=':', alpha=0.4)
ax.grid(True, which='major', linestyle='-', alpha=0.5)
ax.set_xticks([100, 200, 300, 400, 600, 800, 1000])
ax.set_yticks([1, 1.5, 2, 2.5, 3])
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())

ax.set_xlim(200, 1000)
ax.set_ylim(0.95, 3)

ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('plot_dense.png', dpi=300, bbox_inches='tight')
plt.show()