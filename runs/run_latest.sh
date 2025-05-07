#!/bin/bash
set -x  # Print each command before execution

# Default config values (will be used if not set by the calling script)


# Set environment variables
export PYTHONPATH=/home/zihan/CoE:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export MASTER_PORT=29501

torchrun --master_port=$MASTER_PORT --nproc_per_node=4 main.py \
    model.override_config.num_experts_per_tok=4 \
    model.override_config.inner_iter=2 \
    trainer.experiment_name=test-64epts-2iter-4topk