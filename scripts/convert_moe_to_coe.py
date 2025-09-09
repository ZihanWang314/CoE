#!/usr/bin/env python3
"""
Script to convert DeepSeekV2 MoE models to CoE models.

This script implements the transformation:
MoE: g = gate(x), y = experts(g, x)
CoE: g = gate1(x), y1 = experts(g, x)
     y2 = y1 * weight (learnable weight initialized to 0)
     g2 = gate2(y2 + x), y = experts(g2, y2 + x)
     where gate2 is initialized the same as gate1
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import os
from typing import Dict, Any, Optional
import copy

def load_moe_model(model_path: str, device: str = "cpu"):
    """Load a DeepSeekV2 MoE model."""
    print(f"Loading MoE model from {model_path}")
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, config, tokenizer

def create_coe_config(moe_config: Any, inner_iter: int = 2, use_igate: bool = True) -> Any:
    """Create CoE configuration from MoE configuration."""
    print("Creating CoE configuration from MoE configuration")
    
    # Create a copy of the MoE config
    coe_config_dict = moe_config.to_dict()
    
    # Add CoE-specific parameters
    coe_config_dict.update({
        "model_type": "coe_deepseekv2",
        "inner_iter": inner_iter,
        "use_igate": use_igate,
        "inner_residual": True,
        "outer_residual": True,
    })
    
    # Import the CoE config class
    import sys
    sys.path.append('/home/zihan/CoE')
    from config.models.coe_deepseekv2.configuration_coe import CoeConfig
    
    # Create CoE config
    coe_config = CoeConfig(**coe_config_dict)
    
    return coe_config

def convert_moe_layer_to_coe(moe_layer: nn.Module, coe_config: Any) -> nn.Module:
    """Convert a single MoE layer to CoE layer."""
    print("Converting MoE layer to CoE layer")
    
    # Import the CoE modeling classes
    import sys
    sys.path.append('/home/zihan/CoE')
    from config.models.coe_deepseekv2.modeling_coe import CoeMoE, CoeMLP, MoEGate
    
    # Create CoE layer
    coe_layer = CoeMoE(coe_config)
    
    # Copy expert weights
    if hasattr(moe_layer, 'experts') and hasattr(coe_layer, 'experts'):
        print("Copying expert weights")
        for i, (moe_expert, coe_expert) in enumerate(zip(moe_layer.experts, coe_layer.experts)):
            if moe_expert is not None and coe_expert is not None:
                # Copy MLP weights
                if hasattr(moe_expert, 'gate_proj') and hasattr(coe_expert, 'gate_proj'):
                    coe_expert.gate_proj.weight.data.copy_(moe_expert.gate_proj.weight.data)
                if hasattr(moe_expert, 'up_proj') and hasattr(coe_expert, 'up_proj'):
                    coe_expert.up_proj.weight.data.copy_(moe_expert.up_proj.weight.data)
                if hasattr(moe_expert, 'down_proj') and hasattr(coe_expert, 'down_proj'):
                    coe_expert.down_proj.weight.data.copy_(moe_expert.down_proj.weight.data)
    
    # Copy gate weights
    if hasattr(moe_layer, 'gate') and hasattr(coe_layer, 'gate'):
        print("Copying gate weights")
        if coe_config.use_igate:
            # Copy to all gates in the list
            for gate in coe_layer.gate:
                if hasattr(moe_layer.gate, 'weight') and hasattr(gate, 'weight'):
                    gate.weight.data.copy_(moe_layer.gate.weight.data)
        else:
            # Copy to single gate
            if hasattr(moe_layer.gate, 'weight') and hasattr(coe_layer.gate, 'weight'):
                coe_layer.gate.weight.data.copy_(moe_layer.gate.weight.data)
    
    # Copy shared expert weights if they exist
    if hasattr(moe_layer, 'shared_experts') and hasattr(coe_layer, 'shared_experts'):
        if moe_layer.shared_experts is not None and coe_layer.shared_experts is not None:
            print("Copying shared expert weights")
            if hasattr(moe_layer.shared_experts, 'gate_proj') and hasattr(coe_layer.shared_experts, 'gate_proj'):
                coe_layer.shared_experts.gate_proj.weight.data.copy_(moe_layer.shared_experts.gate_proj.weight.data)
            if hasattr(moe_layer.shared_experts, 'up_proj') and hasattr(coe_layer.shared_experts, 'up_proj'):
                coe_layer.shared_experts.up_proj.weight.data.copy_(moe_layer.shared_experts.up_proj.weight.data)
            if hasattr(moe_layer.shared_experts, 'down_proj') and hasattr(coe_layer.shared_experts, 'down_proj'):
                coe_layer.shared_experts.down_proj.weight.data.copy_(moe_layer.shared_experts.down_proj.weight.data)
    
    # Initialize CoE weight to 0 (already done in CoeMoE.__init__)
    print("CoE weight initialized to 0")
    
    return coe_layer

def convert_model_layers(moe_model: nn.Module, coe_config: Any) -> nn.Module:
    """Convert all MoE layers in the model to CoE layers."""
    print("Converting model layers from MoE to CoE")
    
    # Import the CoE modeling classes
    import sys
    sys.path.append('/home/zihan/CoE')
    from config.models.coe_deepseekv2.modeling_coe import CoeModel, CoeForCausalLM
    
    # Create CoE model
    coe_model = CoeForCausalLM(coe_config)
    
    # Copy non-MoE layer weights
    print("Copying non-MoE layer weights")
    for name, moe_param in moe_model.named_parameters():
        if 'experts' not in name and 'gate' not in name:
            # Find corresponding parameter in CoE model
            coe_param = None
            for coe_name, coe_p in coe_model.named_parameters():
                if coe_name == name:
                    coe_param = coe_p
                    break
            
            if coe_param is not None and coe_param.shape == moe_param.shape:
                coe_param.data.copy_(moe_param.data)
                print(f"Copied {name}")
    
    # Convert MoE layers
    print("Converting MoE layers to CoE layers")
    for i, (moe_layer, coe_layer) in enumerate(zip(moe_model.model.layers, coe_model.model.layers)):
        print(f"Converting layer {i}")
        
        # Check if this is an MoE layer
        if hasattr(moe_layer, 'mlp') and hasattr(moe_layer.mlp, 'experts'):
            # Convert MoE layer to CoE layer
            coe_layer.mlp = convert_moe_layer_to_coe(moe_layer.mlp, coe_config)
        else:
            # Copy regular MLP weights
            if hasattr(moe_layer.mlp, 'gate_proj') and hasattr(coe_layer.mlp, 'gate_proj'):
                coe_layer.mlp.gate_proj.weight.data.copy_(moe_layer.mlp.gate_proj.weight.data)
            if hasattr(moe_layer.mlp, 'up_proj') and hasattr(coe_layer.mlp, 'up_proj'):
                coe_layer.mlp.up_proj.weight.data.copy_(moe_layer.mlp.up_proj.weight.data)
            if hasattr(moe_layer.mlp, 'down_proj') and hasattr(coe_layer.mlp, 'down_proj'):
                coe_layer.mlp.down_proj.weight.data.copy_(moe_layer.mlp.down_proj.weight.data)
    
    return coe_model

def save_coe_model(coe_model: nn.Module, coe_config: Any, tokenizer: Any, output_path: str):
    """Save the converted CoE model."""
    print(f"Saving CoE model to {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save model
    coe_model.save_pretrained(output_path)
    
    # Save config
    coe_config.save_pretrained(output_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    
    print(f"CoE model saved successfully to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert DeepSeekV2 MoE model to CoE model")
    parser.add_argument("--moe_model_path", type=str, required=True,
                       help="Path to the MoE model")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save the converted CoE model")
    parser.add_argument("--inner_iter", type=int, default=2,
                       help="Number of inner iterations for CoE (default: 2)")
    parser.add_argument("--use_igate", action="store_true", default=True,
                       help="Use individual gates for each iteration")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use for conversion (default: cpu)")
    
    args = parser.parse_args()
    
    # Load MoE model
    moe_model, moe_config, tokenizer = load_moe_model(args.moe_model_path, args.device)
    
    # Create CoE configuration
    coe_config = create_coe_config(moe_config, args.inner_iter, args.use_igate)
    
    # Convert model
    coe_model = convert_model_layers(moe_model, coe_config)
    
    # Save CoE model
    save_coe_model(coe_model, coe_config, tokenizer, args.output_path)
    
    print("Conversion completed successfully!")

if __name__ == "__main__":
    main()
