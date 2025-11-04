
# Step 1: setup
git clone https://github.com/ZihanWang314/CoE.git
cd CoE
git checkout convert_moe_to_coe
git submodule init
git submodule update
pip install -e verl --no-dependencies

conda create -n coe python=3.12 -y
conda activate coe
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install deepspeed==0.16.9
pip install datasets==3.6.0

Step 2: clone llama factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[metrics]" --no-build-isolation
cd ..


Step 3: initialize checkpoint from a MoE model. This may take several minutes
export PYTHONPATH=$(pwd):$PYTHONPATH
export NEW_MODEL_DIR="./coe_dsv2_lite"
python scripts/convert_moe_to_coe.py --moe_model_path deepseek-ai/deepseek-v2-lite --output_path $NEW_MODEL_DIR --device cuda

cp config/models/coe_deepseekv2/modeling_coe.py $NEW_MODEL_DIR
cp config/models/coe_deepseekv2/configuration_coe.py $NEW_MODEL_DIR

jq '.auto_map={"AutoConfig":"configuration_coe.CoeConfig","AutoModel":"modeling_coe.CoeModel","AutoModelForCausalLM":"modeling_coe.CoeForCausalLM"}' coe_dsv2_lite/config.json > t && mv t ${NEW_MODEL_DIR}/config.json

python scripts/verify_coe_conversion.py --moe_model_path deepseek-ai/deepseek-v2-lite --coe_model_path $NEW_MODEL_DIR # verify if it is correct



Step 4: train with LlamaFactory
```bash
sed -i '/dataset = load_dataset(/,/)/ s/split=dataset_attr.split,/&\
            trust_remote_code=True,/' LLaMA-Factory/src/llamafactory/data/loader.py
jq '. + {"slimpajama-1b": {"hf_hub_url": "iankur/SlimPajama-1B", "columns": {"prompt": "text"}}}' LLaMA-Factory/data/dataset_info.json > tmp && mv tmp LLaMA-Factory/data/dataset_info.json
jq '. + {"slimpajama-627b": {"hf_hub_url": "cerebras/SlimPajama-627B", "columns": {"prompt": "text"}}}' LLaMA-Factory/data/dataset_info.json > tmp && mv tmp LLaMA-Factory/data/dataset_info.json

export PYTHONPATH=$(pwd):$PYTHONPATH
export NEW_MODEL_DIR="./coe_dsv2_lite"
export WANDB_PROJECT="coe"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
llamafactory-cli train third_party/LLaMA-Factory/train_coe.yaml \
    model_name_or_path=$NEW_MODEL_DIR \
    output_dir=outputs/checkpoints/DeepSeek-V2-Lite-CoE-iter8/redpajama \
    resume_from_checkpoint=null \
    dataset=slimpajama-1b \
    stage=pt \
    > log.log 2>&1
    
export WANDB_PROJECT="coe"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
llamafactory-cli train third_party/LLaMA-Factory/train_moe.yaml \
    model_name_or_path=$NEW_MODEL_DIR \
    output_dir=outputs/checkpoints/DeepSeek-V2-Lite/redpajama \
    resume_from_checkpoint=null \
    dataset=slimpajama-1b \
    stage=pt \
    > log.log 2>&1

