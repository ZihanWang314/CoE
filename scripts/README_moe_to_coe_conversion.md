# MoE to CoE Conversion Scripts

This directory contains scripts to convert DeepSeekV2 MoE (Mixture of Experts) models to CoE (Chain of Experts) models.

## Overview

The conversion implements the following transformation:

**Original MoE:**
```
g = gate(x)
y = experts(g, x)
```

**Converted CoE:**
```
g = gate1(x)
y1 = experts(g, x)
y2 = y1 * weight  # learnable weight initialized to 0
g2 = gate2(y2 + x)  # gate2 initialized same as gate1
y = experts(g2, y2 + x)
```

## Files

- `convert_moe_to_coe.py` - Main conversion script
- `verify_coe_conversion.py` - Verification script to test the conversion
- `example_conversion.py` - Example demonstrating the CoE architecture
- `README_moe_to_coe_conversion.md` - This documentation

## Usage

### 1. Convert MoE Model to CoE

```bash
python convert_moe_to_coe.py \
    --moe_model_path /path/to/deepseek-v2-moe \
    --output_path /path/to/converted-coe \
    --inner_iter 2 \
    --use_igate \
    --device cpu
```

**Parameters:**
- `--moe_model_path`: Path to the original DeepSeekV2 MoE model
- `--output_path`: Path where the converted CoE model will be saved
- `--inner_iter`: Number of inner iterations (default: 2)
- `--use_igate`: Use individual gates for each iteration (default: True)
- `--device`: Device to use for conversion (default: cpu)

### 2. Verify Conversion

```bash
python verify_coe_conversion.py \
    --moe_model_path /path/to/deepseek-v2-moe \
    --coe_model_path /path/to/converted-coe \
    --device cpu
```

This script will run several tests to verify:
- CoE weight is initialized to 0
- Gate2 is initialized the same as gate1
- CoE with weight=0 behaves like MoE
- Second iteration is applied when weight != 0

### 3. Example Usage

```bash
python example_conversion.py
```

This demonstrates the CoE architecture components and transformation logic.

## CoE Architecture Changes

The conversion adds the following components to the CoE model:

### Configuration Parameters

- `inner_iter`: Number of inner iterations (default: 1)
- `use_igate`: Use individual gates for each iteration (default: False)
- `inner_residual`: Apply residual connection in inner iterations (default: True)
- `outer_residual`: Apply residual connection in outer iterations (default: True)

### Model Components

- `coe_weight`: Learnable parameter initialized to 0 for the transformation `y2 = y1 * weight`
- Multiple gates when `use_igate=True` and `inner_iter > 1`
- Modified forward pass to implement the CoE transformation

## Key Features

1. **Backward Compatibility**: When `coe_weight = 0`, the CoE model behaves exactly like the original MoE model
2. **Learnable Transformation**: The `coe_weight` parameter allows the model to learn the optimal scaling for the intermediate output
3. **Gate Reuse**: Gate2 is initialized with the same weights as gate1, ensuring consistent behavior
4. **Flexible Architecture**: Supports different numbers of inner iterations and gate configurations

## Verification Tests

The verification script runs the following tests:

1. **Weight Initialization**: Ensures `coe_weight` is initialized to 0
2. **Gate Initialization**: Verifies gate2 is initialized the same as gate1
3. **Parameter Validation**: Checks that `inner_iter` and `use_igate` parameters are set
4. **Forward Pass Equivalence**: Tests that CoE with weight=0 produces the same output as MoE
5. **Second Iteration**: Verifies that the second iteration is applied when weight != 0

## Example Output

```
COE CONVERSION VERIFICATION
============================================================

CoE Weight Initialization:
----------------------------------------
✓ CoE weight in model.model.layers.0.mlp is correctly initialized to 0

Gate Initialization:
----------------------------------------
✓ Gates in model.model.layers.0.mlp are correctly initialized the same

Inner Iteration Parameter:
----------------------------------------
✓ inner_iter parameter is set to 2

Use IGate Parameter:
----------------------------------------
✓ use_igate parameter is set to True

Forward Pass Equivalence:
----------------------------------------
✓ CoE with weight=0 produces equivalent output to MoE

CoE Second Iteration:
----------------------------------------
✓ CoE second iteration is working (different outputs with different weights)

============================================================
VERIFICATION RESULTS: 6/6 tests passed
============================================================
🎉 All tests passed! CoE conversion is working correctly.
```

## Notes

- The conversion preserves all original model weights and only adds the new CoE components
- The `coe_weight` parameter is initialized to 0, ensuring the converted model initially behaves like the original MoE
- During training, the `coe_weight` can be learned to optimize the CoE transformation
- The conversion supports both single and multiple gate configurations
