lm_eval --model hf \
    --model_args pretrained=/shared/storage-01/jiarui14/chain-of-experts/CoE/output/global_step_1000,trust_remote_code=True \
    --tasks hellaswag,piqa,arc_easy \
    --device cuda:8 \
    --batch_size 8