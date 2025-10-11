# # export TRITON_CACHE_DIR='/tmp/triton_cache'
# train_type='lora'
# model_name='/gfshome/LLM-cache/Meta-Llama-3-8B'
# suffix="${model_name##*/}"
# time=$(date +%m%d-%H%M%S)
# llamafactory-cli train examples/train_lora/our_llama3_lora_ppo.yaml \
#     output_dir=output_factory/PPO-${train_type}-${suffix} \
#     overwrite_output_dir=false



# # export TRITON_CACHE_DIR='/tmp/triton_cache'
# train_type='lora'
# model_name='/gfshome/LLM-cache/Mistral-7B-v0.1'
# suffix="${model_name##*/}"
# time=$(date +%m%d-%H%M%S)
# llamafactory-cli train examples/train_lora/our_mistral_lora_ppo.yaml \
#     output_dir=output_factory/PPO-${train_type}-${suffix} \
#     overwrite_output_dir=false



# # REVIEW: 09.10新尝试
# export TRITON_CACHE_DIR='/tmp/triton_cache'
# train_type='lora'
# model_name='/gfshome/LLM-cache/Mistral-7B-v0.1'
# suffix="${model_name##*/}"
# time=$(date +%m%d-%H%M%S)
# llamafactory-cli train examples/train_lora/our_mistral_lora_ppo.yaml \
#     output_dir=output_factory/PPOmixRAG-${train_type}-${suffix} \
#     overwrite_output_dir=false \
#     num_train_epochs=1.0
# # update: 传入的merged model忘了换...白训了。



#==================== Training Scaling K=12 ====================
# # export TRITON_CACHE_DIR='/tmp/triton_cache'
# train_type='lora'
# model_name='/gfshome/LLM-cache/Meta-Llama-3-8B'
# suffix="${model_name##*/}"
# time=$(date +%m%d-%H%M%S)
# llamafactory-cli train examples/train_lora/our_llama3_lora_ppo.yaml \
#     output_dir=output_factory/PPO-${train_type}-${suffix} \
#     overwrite_output_dir=false

# num_train_epochs: 2.0 --> 0.45 --> 0.3


#==================== Training Scaling K=10, 8 ====================

# train_type='lora'
# model_name='/gfshome/LLM-cache/Meta-Llama-3-8B'
# suffix="${model_name##*/}"
# time=$(date +%m%d-%H%M%S)
# llamafactory-cli train examples/train_lora/our_llama3_lora_ppo.yaml \
#     overwrite_output_dir=false \
#     output_dir=output_factory/K10-PPO-${train_type}-${suffix} \
#     dataset=our_llama3_sft_dataset_k10 \
#     num_train_epochs=0.3 \
#     model_name_or_path=output_merged/K10-SFT-lora-Meta-Llama-3-8B \
#     reward_model=output_factory/K10-RM-lora-Meta-Llama-3-8B/checkpoint-100


train_type='lora'
model_name='/gfshome/LLM-cache/Meta-Llama-3-8B'
suffix="${model_name##*/}"
time=$(date +%m%d-%H%M%S)
llamafactory-cli train examples/train_lora/our_llama3_lora_ppo.yaml \
    overwrite_output_dir=false \
    output_dir=output_factory/K8-PPO-${train_type}-${suffix} \
    dataset=our_llama3_sft_dataset_k8 \
    num_train_epochs=0.3 \
    model_name_or_path=output_merged/K8-SFT-lora-Meta-Llama-3-8B \
    reward_model=output_factory/K8-RM-lora-Meta-Llama-3-8B/checkpoint-100