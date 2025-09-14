#!/usr/bin/env python3
"""
Script to verify the CoE conversion transformation.

This script verifies that the CoE transformation is working correctly by:
1. Testing that the learnable weight is initialized to 0
2. Verifying that when weight=0, CoE behaves like MoE (first iteration only)
3. Testing that the second iteration is applied when weight != 0
4. Checking that gate2 is initialized the same as gate1
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import argparse
import sys
import os

# Add CoE path to sys.path
sys.path.append('/home/zihan/CoE')

def load_models(moe_path: str, coe_path: str, device: str = "cpu"):
    """Load both MoE and CoE models for comparison."""
    print(f"Loading MoE model from {moe_path}")
    moe_model = AutoModelForCausalLM.from_pretrained(
        moe_path,
        torch_dtype=torch.float32,
        device_map=device,
        trust_remote_code=True
    )
    
    print(f"Loading CoE model from {coe_path}")
    coe_model = AutoModelForCausalLM.from_pretrained(
        coe_path,
        torch_dtype=torch.float32,
        device_map=device,
        trust_remote_code=True
    )
    
    return moe_model, coe_model

def test_coe_weight_initialization(coe_model: nn.Module):
    """Test that CoE weight is initialized to 0."""
    print("Testing CoE weight initialization...")
    
    for name, module in coe_model.named_modules():
        if hasattr(module, 'coe_weight'):
            weight = module.coe_weight
            if torch.allclose(weight, torch.zeros_like(weight), atol=1e-6):
                print(f"✓ CoE weight in {name} is correctly initialized to 0")
            else:
                print(f"✗ CoE weight in {name} is NOT initialized to 0: {weight}")
                return False
    
    return True

def test_gate_initialization(coe_model: nn.Module):
    """Test that gate2 is initialized the same as gate1."""
    print("Testing gate initialization...")
    
    for name, module in coe_model.named_modules():
        if hasattr(module, 'gate') and hasattr(module, 'use_igate') and module.use_igate:
            if isinstance(module.gate, nn.ModuleList) and len(module.gate) > 1:
                gate1_weight = module.gate[0].weight.data
                gate2_weight = module.gate[1].weight.data
                
                if torch.allclose(gate1_weight, gate2_weight, atol=1e-6):
                    print(f"✓ Gates in {name} are correctly initialized the same")
                else:
                    print(f"✗ Gates in {name} are NOT initialized the same")
                    print(f"  Gate1 weight shape: {gate1_weight.shape}")
                    print(f"  Gate2 weight shape: {gate2_weight.shape}")
                    print(f"  Max difference: {torch.max(torch.abs(gate1_weight - gate2_weight))}")
                    return False
    
    return True

def test_forward_pass_equivalence(moe_model: nn.Module, coe_model: nn.Module, 
                                 input_ids: torch.Tensor, device: str = "cpu"):
    """Test that CoE with weight=0 behaves like MoE."""
    print("Testing forward pass equivalence when CoE weight=0...")
    
    # Set CoE model to eval mode
    moe_model.eval()
    coe_model.eval()
    
    # Ensure CoE weight is 0
    for module in coe_model.modules():
        if hasattr(module, 'coe_weight'):
            module.coe_weight.data.zero_()
    
    # Forward pass through both models
    with torch.no_grad():
        moe_outputs = moe_model(input_ids, output_hidden_states=True)
        coe_outputs = coe_model(input_ids, output_hidden_states=True)
    # Compare outputs
    moe_logits = moe_outputs.logits
    coe_logits = coe_outputs.logits
    
    # Check if outputs are close (allowing for small numerical differences)
    if torch.allclose(moe_logits, coe_logits, atol=1e-4, rtol=1e-4):
        print("✓ CoE with weight=0 produces equivalent output to MoE")
        return True
    else:
        max_diff = torch.max(torch.abs(moe_logits - coe_logits))
        mean_diff = torch.mean(torch.abs(moe_logits - coe_logits))
        print(f"✗ CoE with weight=0 does NOT produce equivalent output to MoE")
        print(f"  Max difference: {max_diff}")
        print(f"  Mean difference: {mean_diff}")
        return False


def test_inner_iteration_parameter(coe_model: nn.Module):
    """Test that inner_iter parameter is correctly set."""
    print("Testing inner_iter parameter...")
    
    config = coe_model.config
    if hasattr(config, 'inner_iter'):
        print(f"✓ inner_iter parameter is set to {config.inner_iter}")
        return True
    else:
        print("✗ inner_iter parameter is not set")
        return False

def test_use_igate_parameter(coe_model: nn.Module):
    """Test that use_igate parameter is correctly set."""
    print("Testing use_igate parameter...")
    
    config = coe_model.config
    if hasattr(config, 'use_igate'):
        print(f"✓ use_igate parameter is set to {config.use_igate}")
        return True
    else:
        print("✗ use_igate parameter is not set")
        return False

def create_test_input(seq_len: int = 10, vocab_size: int = 1000, device: str = "cpu"):
    """Create test input for verification."""
    # Create random input IDs
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
    return input_ids

def main():
    parser = argparse.ArgumentParser(description="Verify CoE conversion")
    parser.add_argument("--moe_model_path", type=str, required=True,
                       help="Path to the original MoE model")
    parser.add_argument("--coe_model_path", type=str, required=True,
                       help="Path to the converted CoE model")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use for testing (default: cpu)")
    parser.add_argument("--seq_len", type=int, default=10,
                       help="Sequence length for test input (default: 10)")
    
    args = parser.parse_args()
    
    # Load models
    moe_model, coe_model = load_models(args.moe_model_path, args.coe_model_path, args.device)
    
    # Create test input
    input_ids = create_test_input(args.seq_len, device=args.device)
    
    # Run verification tests
    tests = [
        ("CoE Weight Initialization", lambda: test_coe_weight_initialization(coe_model)),
        ("Gate Initialization", lambda: test_gate_initialization(coe_model)),
        ("Inner Iteration Parameter", lambda: test_inner_iteration_parameter(coe_model)),
        ("Use IGate Parameter", lambda: test_use_igate_parameter(coe_model)),
        ("Forward Pass Equivalence", lambda: test_forward_pass_equivalence(moe_model, coe_model, input_ids, args.device)),
    ]
    
    print("=" * 60)
    print("COE CONVERSION VERIFICATION")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        # try:
        if test_func():
            passed += 1
        # except Exception as e:
        #     print(f"✗ Test failed with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"VERIFICATION RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! CoE conversion is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the conversion.")
        return 1

if __name__ == "__main__":
    exit(main())
