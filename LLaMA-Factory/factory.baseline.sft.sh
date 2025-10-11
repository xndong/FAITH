# export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR='/tmp/triton_cache'
train_type='lora'
model_name='/gfshome/LLM-cache/Mistral-7B-v0.1'
suffix="${model_name##*/}"
time=$(date +%m%d-%H%M%S)
llamafactory-cli train examples/train_lora/our_mistral_lora_sft.yaml \
    model_name_or_path=${model_name} \
    dataset=baseline_sft_dataset \
    output_dir=output_factory/BASELINE-SFT-${train_type}-${suffix} \
    overwrite_output_dir=false