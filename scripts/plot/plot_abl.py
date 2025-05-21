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
data = pd.read_csv('data5.tsv', sep='\t')

data.columns = [col.strip() for col in data.columns]

steps = np.arange(10, (len(data) + 1) * 10, 10)

methods = [
    'dsv2_coe_3-fl_64ept-8tpk-1itr - val/loss',
    'dsv2_coe_3-cc_64ept-8tpk-2itr - val/loss',    
    'dsv2_coe_3-ab_64ept-8tpk-2itr-noig - val/loss', 
    'dsv2_coe_3-ab_64ept-8tpk-2itr-ore - val/loss',  
]

labels = [
    'MoE (K=8, N=64)',
    'CoE (C=2, K=4, N=64)',
    'w/o Independent Gate',
    'w/o Inner Residual',
]

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#4daf4a']
markers = ['o', 's', 'D', '^']
linestyles = ['-', '-', ':', '-.']

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
# ax.set_title('Ablation Study: Independent Gate & Inner Residual', fontweight='bold', fontsize=16)
ax.legend(frameon=True, fontsize=12, framealpha=0.7, edgecolor='#333333', loc='upper right')
ax.grid(True, which='minor', linestyle=':', alpha=0.4)
ax.grid(True, which='major', linestyle='-', alpha=0.5)

ax.set_xticks([100, 200, 300, 400, 600, 800, 1000])
ax.set_yticks([1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3])
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.xaxis.get_major_formatter().set_scientific(False)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.yaxis.get_major_formatter().set_scientific(False)
# Adjust axis limits to focus on the important part
plt.xlim(190, 1000)
plt.ylim(1, 3)


ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('plot_abl.png', dpi=300, bbox_inches='tight')
plt.show()