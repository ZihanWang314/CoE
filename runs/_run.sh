#!/bin/bash
set -x  # Print each command before execution

# Default config values (will be used if not set by the calling script)
GENERAL_NAME=${GENERAL_NAME:-"run"}
CUDA_DEVICE=${CUDA_DEVICE:-0}
NUM_GPUS=${NUM_GPUS:-1}
MASTER_PORT=${MASTER_PORT:-29500}
CONFIGS=${CONFIGS:-()}
PROJECT_NAME=${PROJECT_NAME:-"metamathqa-sft"}
TRAIN_FILES=${TRAIN_FILES:-"data/metamathqa/train.parquet"}
VAL_FILES=${VAL_FILES:-"data/metamathqa/test.parquet"}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-32}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1000}
VALIDATION_INTERVAL_STEPS=${VALIDATION_INTERVAL_STEPS:-10}
TOTAL_VALIDATION_COUNT=${TOTAL_VALIDATION_COUNT:-100}
MAX_LENGTH=${MAX_LENGTH:-512}
TEXT_KEYS=${TEXT_KEYS:-"['query','response']"}
LOGGER=${LOGGER:-"['console','wandb']"}
MODEL_PATH=${MODEL_PATH:-"config/models/coe_deepseekv2"}
ATTN_IMPL=${ATTN_IMPL:-"flash_attention_2"}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-"output"}
DEFAULT_HDFS_DIR=${DEFAULT_HDFS_DIR:-"null"}
TRUNCATION=${TRUNCATION:-"right"}
N_SHARED_EXPERTS=${N_SHARED_EXPERTS:-1}
LR_SCHEDULER=${LR_SCHEDULER:-"constant"}

# arguments that may be overridden by the calling script
OUTER_RESIDUAL=${OUTER_RESIDUAL:-false}
INNER_RESIDUAL=${INNER_RESIDUAL:-true}
USE_IGATE=${USE_IGATE:-true}

# Set environment variables
export PYTHONPATH=/home/zihan/CoE:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
export MASTER_PORT=$MASTER_PORT

# Build base command line arguments
base_args=(
    "trainer.project_name=$PROJECT_NAME"
    "data.train_files=$TRAIN_FILES"
    "data.val_files=$VAL_FILES"
    "data.truncation=$TRUNCATION"
    "data.max_length=$MAX_LENGTH"
    "+data.text_keys=$TEXT_KEYS"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "+fsdp=true"
    "model.partial_pretrain=$MODEL_PATH"
    "+model.from_config=true"
    "+model.override_config._attn_implementation=$ATTN_IMPL"
    "+model.override_config.outer_residual=$OUTER_RESIDUAL"
    "+model.override_config.inner_residual=$INNER_RESIDUAL"
    "+model.override_config.use_igate=$USE_IGATE"
    "+model.override_config.n_shared_experts=$N_SHARED_EXPERTS"
    # "trainer.default_local_dir=$DEFAULT_LOCAL_DIR"
    "trainer.total_epochs=null"
    "trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
    "trainer.validation_interval_steps=$VALIDATION_INTERVAL_STEPS"
    "trainer.total_validation_count=$TOTAL_VALIDATION_COUNT"
    "+trainer.n_gpus_per_node=$NUM_GPUS"
    "trainer.logger=$LOGGER"
    "trainer.default_hdfs_dir=$DEFAULT_HDFS_DIR"
    "optim.lr_scheduler=$LR_SCHEDULER"
)

# Add any additional base args defined in the calling script
if [[ -n "$EXTRA_ARGS" ]]; then
    IFS=' ' read -r -a extra_args_array <<< "$EXTRA_ARGS"
    base_args+=("${extra_args_array[@]}")
fi

# Run each configuration
for config in "${CONFIGS[@]}"; do
    # First split by ':' to get suffix and the rest of the config
    IFS=':' read -r suffix rest_of_config <<< "$config"
    
    # Build command
    cmd="torchrun --standalone --nnodes=1 --master_port=$MASTER_PORT --nproc_per_node=$NUM_GPUS main.py"
    
    # Add base arguments
    for arg in "${base_args[@]}"; do
        cmd+=" $arg"
    done
    
    # Add experiment name
    cmd+=" trainer.experiment_name=$GENERAL_NAME-$suffix"
    cmd+=" trainer.default_local_dir=checkpoints/$GENERAL_NAME-$suffix"
    
    # Add micro batch size
    cmd+=" data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU"
    echo "Batch size per GPU: $MICRO_BATCH_SIZE_PER_GPU"
    
    # Parse the rest of the config which is in key=value format
    # Special case: if we find num_experts, calculate num_routed_experts
    num_experts_found=false
    
    # Check if rest_of_config is not empty
    if [[ -n "$rest_of_config" ]]; then
        IFS=':' read -r -a config_entries <<< "$rest_of_config"
        for entry in "${config_entries[@]}"; do
            # Split into key-value
            IFS='=' read -r key value <<< "$entry"
            
            # Special case for num_experts to calculate num_routed_experts
            if [[ "$key" == "num_experts" ]]; then
                num_experts_found=true
                num_routed_experts=$((value - N_SHARED_EXPERTS))
                cmd+=" +model.override_config.n_routed_experts=$num_routed_experts"
            fi
            
            # Add to command
            # if key is in above, use ++, otherwise use +
            if [[ "$key" == "outer_residual" || "$key" == "inner_residual" || "$key" == "use_igate" ]]; then
                cmd+=" ++model.override_config.$key=$value"
            else
                cmd+=" +model.override_config.$key=$value"
            fi
        done
    fi
    
    # Process any additional model overrides from the calling script
    if [[ -n "$MODEL_OVERRIDES" ]]; then
        IFS=' ' read -r -a override_array <<< "$MODEL_OVERRIDES"
        for override in "${override_array[@]}"; do
            IFS='=' read -r key value <<< "$override"
            cmd+=" +model.override_config.$key=$value"
        done
    fi

    # Add any additional arguments passed to this script
    cmd+=" $@"
    
    # Execute command
    echo "Running experiment: $GENERAL_NAME-$suffix"
    eval $cmd
done