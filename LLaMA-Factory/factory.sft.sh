# # export CUDA_VISIBLE_DEVICES=0
# export TRITON_CACHE_DIR='/tmp/triton_cache'
# train_type='lora'
# model_name='/gfshome/LLM-cache/Meta-Llama-3-8B'
# suffix="${model_name##*/}"
# time=$(date +%m%d-%H%M%S)
# llamafactory-cli train examples/train_lora/our_llama3_lora_sft.yaml \
#     model_name_or_path=${model_name} \
#     output_dir=output_factory/SFT-${train_type}-${suffix} \
#     overwrite_output_dir=false



# # export CUDA_VISIBLE_DEVICES=0
# export TRITON_CACHE_DIR='/tmp/triton_cache'
# train_type='lora'
# model_name='/gfshome/LLM-cache/Mistral-7B-v0.1'
# suffix="${model_name##*/}"
# time=$(date +%m%d-%H%M%S)
# llamafactory-cli train examples/train_lora/our_mistral_lora_sft.yaml \
#     model_name_or_path=${model_name} \
#     output_dir=output_factory/SFT-${train_type}-${suffix} \
#     overwrite_output_dir=false



# # REVIEW: 09.10新尝试
# # export CUDA_VISIBLE_DEVICES=0
# export TRITON_CACHE_DIR='/tmp/triton_cache'
# train_type='lora'
# model_name='/gfshome/LLM-cache/Mistral-7B-v0.1'
# suffix="${model_name##*/}"
# time=$(date +%m%d-%H%M%S)
# llamafactory-cli train examples/train_lora/our_mistral_lora_sft.yaml \
#     model_name_or_path=${model_name} \
#     dataset=our_sft_dataset,chatqa_train_squad2.0 \
#     output_dir=output_factory/SFTmixRAG-${train_type}-${suffix} \
#     overwrite_output_dir=false

#==================== Training Scaling K=12 ====================

# # export CUDA_VISIBLE_DEVICES=0
# export TRITON_CACHE_DIR='/tmp/triton_cache'
# train_type='lora'
# model_name='/gfshome/LLM-cache/Meta-Llama-3-8B'
# suffix="${model_name##*/}"
# time=$(date +%m%d-%H%M%S)
# llamafactory-cli train examples/train_lora/our_llama3_lora_sft.yaml \
#     model_name_or_path=${model_name} \
#     output_dir=output_factory/SFT-${train_type}-${suffix} \
#     overwrite_output_dir=false


#==================== Training Scaling K=10, 8 ====================


export TRITON_CACHE_DIR='/tmp/triton_cache'
train_type='lora'
model_name='/gfshome/LLM-cache/Meta-Llama-3-8B'
suffix="${model_name##*/}"
time=$(date +%m%d-%H%M%S)
llamafactory-cli train examples/train_lora/our_llama3_lora_sft.yaml \
    model_name_or_path=${model_name} \
    overwrite_output_dir=false \
    output_dir=output_factory/K10-SFT-${train_type}-${suffix} \
    dataset=our_llama3_sft_dataset_k10

export TRITON_CACHE_DIR='/tmp/triton_cache'
train_type='lora'
model_name='/gfshome/LLM-cache/Meta-Llama-3-8B'
suffix="${model_name##*/}"
time=$(date +%m%d-%H%M%S)
llamafactory-cli train examples/train_lora/our_llama3_lora_sft.yaml \
    model_name_or_path=${model_name} \
    overwrite_output_dir=false \
    output_dir=output_factory/K8-SFT-${train_type}-${suffix} \
    dataset=our_llama3_sft_dataset_k8