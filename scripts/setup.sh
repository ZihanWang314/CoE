#!/bin/bash

# Exit on error
set -e

# Function to check if CUDA is available
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA GPU detected"
        return 0
    else
        echo "No CUDA GPU detected"
        return 1
    fi
}

# Function to check if conda is available
check_conda() {
    if command -v conda &> /dev/null; then
        echo "Conda is available"
        return 0
    else
        echo "Conda is not installed. Please install Conda first."
        return 1
    fi
}

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print step with color
print_step() {
    echo -e "${BLUE}[Step] ${1}${NC}"
}

# Main installation process
main() {
    # Check prerequisites
    check_conda || exit 1
    
    # Create and activate conda environment
    print_step "Creating conda environment 'coe' with Python 3.9..."
    conda create -n coe python=3.9 -y
    
    # Need to source conda for script environment
    eval "$(conda shell.bash hook)"
    conda activate coe
    
    # Clone repository
    # print_step "Cloning coe repository..."
    # git clone git@github.com:ZihanWang314/coe.git
    # cd coe
    
    # Install package in editable mode
    print_step "setting up verl..."
    git submodule init
    git submodule update
    cd verl
    pip install -e .
    cd ..

    print_step "downloading dataset..."
    python scripts/download_dataset.py

    # Install PyTorch with CUDA if available
    if check_cuda; then
        print_step "CUDA detected, checking CUDA version..."
        
        if command -v nvcc &> /dev/null; then
            nvcc_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
            nvcc_major=$(echo $nvcc_version | cut -d. -f1)
            nvcc_minor=$(echo $nvcc_version | cut -d. -f2)
            
            print_step "Found NVCC version: $nvcc_version"
            
            if [[ "$nvcc_major" -gt 12 || ("$nvcc_major" -eq 12 && "$nvcc_minor" -ge 1) ]]; then
                print_step "CUDA $nvcc_version is already installed and meets requirements (>=12.1)"
                export CUDA_HOME=${CUDA_HOME:-$(dirname $(dirname $(which nvcc)))}
            else
                print_step "CUDA version < 12.1, installing CUDA toolkit 12.1..."
                conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
                export CUDA_HOME=$CONDA_PREFIX
            fi
        else
            print_step "NVCC not found, installing CUDA toolkit 12.1..."
            conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
            export CUDA_HOME=$CONDA_PREFIX
        fi
        
        print_step "Installing PyTorch with CUDA support..."
        pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
        
        print_step "Installing flash-attention..."
        pip3 install flash-attn --no-build-isolation
    else
        print_step "Installing PyTorch without CUDA support..."
        pip install torch==2.4.0
    fi
    
    # Install remaining requirements
    print_step "Installing additional requirements..."
    pip install -r requirements.txt
    
    echo -e "${GREEN}Installation completed successfully!${NC}"
    echo "To activate the environment, run: conda activate coe"
}

# Run main installation
main