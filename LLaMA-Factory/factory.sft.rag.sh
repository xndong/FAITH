
# # export CUDA_VISIBLE_DEVICES=0
# export TRITON_CACHE_DIR='/tmp/triton_cache'
# train_type='lora'
# model_name='/gfshome/LLM-cache/Mistral-7B-v0.1'
# suffix="${model_name##*/}"
# time=$(date +%m%d-%H%M%S)
# llamafactory-cli train examples/train_lora/our_rag_mistral_lora_sft.yaml \
#     model_name_or_path=${model_name} \
#     output_dir=output_factory/RAG-SFT-${train_type}-${suffix} \
#     overwrite_output_dir=false



#REVIEW: 09.10 新尝试
# export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR='/tmp/triton_cache'
train_type='lora'
model_name='/gfshome/LLM-cache/Mistral-7B-v0.1'
suffix="${model_name##*/}"
time=$(date +%m%d-%H%M%S)
llamafactory-cli train examples/train_lora/our_rag_mistral_lora_sft.yaml \
    model_name_or_path=${model_name} \
    output_dir=output_factory/RAG-mix5k-SFT-${train_type}-${suffix} \
    overwrite_output_dir=false \
    dataset=rag_mistral_dataset,chatqa_train_squad2.0_num5k \
    num_train_epochs=1