import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.models.coe_deepseekv2.modeling_coe import CoeConfig, CoeForCausalLM
from datasets import load_dataset
from matplotlib.colors import LogNorm

from tqdm import tqdm

output_path = "outputs/figure_coe"
# model_name = "chain-of-experts/64ept-8tpk-1itr-slimpajama-lr2e-5-10k"
# model_type = "moe"

model_name = "chain-of-experts/64ept-4tpk-2itr-1SharedExp-metamathqa-2k-lr5e-5-bs256"
model_type = "coe"

# model_name = "chain-of-experts/64ept-4tpk-2itr-metamathqa-10k"
# model_type = "coe"

# model_name = "chain-of-experts/64ept-4tpk-2itr-slimpajama-lr2e-5-10k"
# model_type = "coe"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# data = pd.read_parquet("data/gsm8k/test.parquet")
# data_type = "gsm8k"

# data = load_dataset("iankur/SlimPajama-100M", split="train")
# data_type = "slimpajama"

data = load_dataset("meta-math/MetaMathQA-40K", split="train")
data_type = "metamathqa"


os.makedirs(output_path, exist_ok=True)

# Load model and set up for routing logit saving
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32, trust_remote_code=True
)
model.save_pretrained(f"outputs/{model_name}")

config = model.config
config.save_routing_logits = True
model = CoeForCausalLM(config)
state_dict = load_file(f"outputs/{model_name}/model.safetensors")
model.load_state_dict(state_dict)
model.eval()
model.to("cuda")

# Prepare accumulation containers
selected_layers = [0, 1, config.num_hidden_layers - 2, config.num_hidden_layers - 1]
matrix_dict = {layer: torch.zeros(63, 63) for layer in selected_layers}
expert_count = {layer: {_iter: torch.zeros(63) for _iter in range(2)} for layer in selected_layers}
# Accumulate over 100 samples
for idx in tqdm(range(1000)):
    
    if data_type == "gsm8k":
        instance = data.iloc[idx]
        text = "question: " + instance["question"] + " answer: " + instance["answer"]
    elif data_type == "slimpajama":
        instance = data[idx]
        text = instance["text"]
    elif data_type == "metamathqa":
        instance = data[idx]
        text = "question: " + instance["query"] + " answer: " + instance["response"]
    input_ids = tokenizer.encode(text, return_tensors="pt")[:, :1024].to("cuda")
    
    with torch.no_grad():
        _ = model(input_ids)

    if model_type == "coe":
        for layer in selected_layers:
            iter0_path = f"outputs/routing_logits/layer_{layer}/iter_0_topk_idx.pt"
            iter1_path = f"outputs/routing_logits/layer_{layer}/iter_1_topk_idx.pt"
            iter0_topk_idx = torch.load(iter0_path).cpu().numpy()
            iter1_topk_idx = torch.load(iter1_path).cpu().numpy()

            for topk0, topk1 in zip(iter0_topk_idx, iter1_topk_idx):
                matrix_dict[layer][topk0, topk1] += 1

            for topk0 in iter0_topk_idx:
                expert_count[layer][0][topk0] += 1
            for topk1 in iter1_topk_idx:
                expert_count[layer][1][topk1] += 1

    elif model_type == "moe":
        for layer in selected_layers:
            iter0_path = f"outputs/routing_logits/layer_{layer}/iter_0_topk_idx.pt"
            iter0_topk_idx = torch.load(iter0_path).cpu().numpy()

            for topk0 in iter0_topk_idx:
                matrix_dict[layer][topk0, topk0] += 1



fig, axes = plt.subplots(1, len(selected_layers), figsize=(3 * len(selected_layers), 3))

# Plot and save figures
for layer in selected_layers:
    matrix = matrix_dict[layer]
    # remove all-zero rows and columns
    used_experts = (matrix.sum(dim=1) > 0) | (matrix.sum(dim=0) > 0)
    matrix = matrix[used_experts, :][:, used_experts]
    matrix = matrix + 1
    ax = axes[layer]
    im = ax.imshow(matrix.numpy(), cmap='viridis', norm=LogNorm(vmin=1, vmax=matrix.max().item() * 10))
    ax.set_title(f"Layer {layer}")
    ax.set_xlabel("Expert in Iter1")
    if layer == 0:
        ax.set_ylabel("Expert in Iter0")

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height] in figure coords
fig.colorbar(im, cax=cbar_ax)
fig.tight_layout(rect=[0, 0, 0.91, 1])
plt.savefig(f"{output_path}/layer_all_routing_matrix.png", dpi=300)
plt.close()

if model_type == "coe":
    for layer in selected_layers:
        plot_first_iter = expert_count[layer][0]  # shape: (64,)
        plot_second_iter = expert_count[layer][1]  # shape: (64,)

        x = np.arange(len(plot_first_iter))  # 0 to 63
        width = 0.4

        plt.figure(figsize=(10, 5))
        plt.bar(x - width/2, plot_first_iter, width=width, label='First Iter')
        plt.bar(x + width/2, plot_second_iter, width=width, label='Second Iter')
        plt.xlabel("Expert Index")
        plt.ylabel("Usage Count")
        plt.title(f"Expert Usage Distribution (Layer {layer})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_path}/layer_{layer}_expert_count.png")
        plt.close()
