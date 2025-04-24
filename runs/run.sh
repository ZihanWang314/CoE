GPUS=(8 9)
export NUM_GPUS=${#GPUS[@]}
export CUDA_DEVICE=$(IFS=,; echo "${GPUS[*]}")
export MASTER_PORT=29500

# Define configuration combinations
# Format: experiment_suffix:sparsity:granularity:topk:num_hidden_layers:use_igate

export GENERAL_NAME="dsv2_coe"
export CONFIGS=(
    "main_64ept-8tpk-1itr-debug:num_experts=64:num_experts_per_tok=8:inner_iter=1"
    # "main_64ept-4tpk-2itr:num_experts=64:num_experts_per_tok=4:inner_iter=2"

    #########################################################################
    ### for below analysis experiments, please run it based on your own needs
    #########################################################################

    # "cc_64ept-16tpk-1itr:num_experts=64:num_experts_per_tok=16:inner_iter=1"
    # "cc_64ept-24tpk-1itr:num_experts=64:num_experts_per_tok=24:inner_iter=1"
    # "cc_64ept-8tpk-2itr:num_experts=64:num_experts_per_tok=8:inner_iter=2"

    # "mm_48ept-4tpk-2itr:num_experts=48:num_experts_per_tok=4:inner_iter=2"
    # "mm_48ept-8tpk-1itr:num_experts=48:num_experts_per_tok=8:inner_iter=1"

    # "ab_64ept-8tpk-2itr-ore:num_experts=64:num_experts_per_tok=8:inner_iter=2:outer_residual=true"
    # "ab_64ept-8tpk-2itr-noig:num_experts=64:num_experts_per_tok=8:inner_iter=2:outer_residual=true:use_igate=false"

    # "ds_8ept-8tpk-2itr:num_experts=8:num_experts_per_tok=8:inner_iter=2"
    # "ds_8ept-8tpk-1itr:num_experts=8:num_experts_per_tok=8:inner_iter=1"

    # "ly_64ept-8tpk-1itr-8lyr:num_experts=64:num_experts_per_tok=8:inner_iter=1:num_hidden_layers=8"
    # "ly_64ept-8tpk-1itr-12lyr:num_experts=64:num_experts_per_tok=8:inner_iter=1:num_hidden_layers=12"
)

metamathqa_train_path=data/metamathqa/train.parquet
slimpajama_train_path=data/SlimPajama-6B/train.parquet
openr1math_train_path=data/OpenR1-Math-220k/train.parquet
gsm8k_train_path=data/gsm8k/train.parquet
gsm8k_test_path=data/gsm8k/test.parquet

# 训练集文件路径
# export TRAIN_FILES="\"['$metamathqa_train_path', '$slimpajama_train_path', '$openr1math_train_path']\""
export TRAIN_FILES="data/metamathqa/test.parquet"
# export VAL_FILES="\"['$gsm8k_test_path']\""
export TRAIN_BATCH_SIZE=64
export MICRO_BATCH_SIZE_PER_GPU=64
export LR_SCHEDULER="constant"
export N_SHARED_EXPERTS=1
export MAX_LENGTH=1024
export TOTAL_TRAINING_STEPS=100000 # total tokens: 100k * 64 * 1024 = 6B
export LOGGER="['console']"

# Add any extra configurations or overrides

# For example, to override or add model configuration:
# export MODEL_OVERRIDES="use_decoder_only_format=true dropout=0.1"

# Add any extra arguments to the base command
# export EXTRA_ARGS="+trainer.checkpoint_interval_steps=50 +trainer.seed=42"

# Run the common script
source "$(dirname "$0")/_run.sh" "$@"