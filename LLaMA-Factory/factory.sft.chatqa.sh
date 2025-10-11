#? 下面3个阶段的yaml中template改为llama3，方便跑DTA的evaluation，但是实际训练时忘了改回来。
#? Update: 0905加入了template参数,但没有重新训练一遍，见最下面注释。

#! /bin/bash
# set -e

# #* ==================== Stage 1 ====================
# # export CUDA_VISIBLE_DEVICES=0
# export TRITON_CACHE_DIR='/tmp/triton_cache'
# train_type='lora'
# model_name='/gfshome/LLM-cache/Mistral-7B-v0.1'
# suffix="${model_name##*/}"
# time=$(date +%m%d-%H%M%S)
# llamafactory-cli train examples/train_lora/our_mistral_lora_chatqa_stage1_sft.yaml \
#     model_name_or_path=${model_name} \
#     output_dir=output_factory/ChatQA-SFT-${train_type}-${suffix} \
#     overwrite_output_dir=false \


# #* =============== Merge Lora adapter ===============
# llamafactory-cli export examples/merge_lora/mistral_lora_sft.chatqa.yaml \
#     adapter_name_or_path=output_factory/ChatQA-SFT-lora-Mistral-7B-v0.1/checkpoint-390 \
#     export_dir=output_merged/ChatQA-SFT-lora-Mistral-7B-v0.1 \


# #* ==================== Stage 2 =====================
# # export CUDA_VISIBLE_DEVICES=0
# export TRITON_CACHE_DIR='/tmp/triton_cache'
# train_type='lora'
# model_name='/gfshome/LLM-cache/Mistral-7B-v0.1'
# suffix="${model_name##*/}"
# time=$(date +%m%d-%H%M%S)
# llamafactory-cli train examples/train_lora/our_mistral_lora_chatqa_stage2_sft.yaml \
#     model_name_or_path=output_merged/ChatQA-SFT-lora-Mistral-7B-v0.1 \
#     output_dir=output_factory/ChatQA-SFT-stage2-${train_type}-${suffix} \
#     overwrite_output_dir=false \


# #* =============== Merge Lora adapter ===============
# llamafactory-cli export examples/merge_lora/mistral_lora_sft.chatqa.stage2.yaml \
#     adapter_name_or_path=output_factory/ChatQA-SFT-stage2-lora-Mistral-7B-v0.1/checkpoint-464 \
#     # export_dir=output_merged/ChatQA-SFT-stage2-lora-Mistral-7B-v0.1



















#################################################################################################################
#* bash factory.sft.chatqa.sh llama3
#* bash factory.sft.chatqa.sh default

# template="$1" # default
# #* ==================== Stage 1 ====================
# # export CUDA_VISIBLE_DEVICES=0
# export TRITON_CACHE_DIR='/tmp/triton_cache'
# train_type='lora'
# model_name='/gfshome/LLM-cache/Mistral-7B-v0.1'
# suffix="${model_name##*/}"
# time=$(date +%m%d-%H%M%S)
# llamafactory-cli train examples/train_lora/our_mistral_lora_chatqa_stage1_sft.yaml \
#     model_name_or_path=${model_name} \
#     output_dir=output_factory/ChatQA-tmpl-${template}-SFT-${train_type}-${suffix} \
#     overwrite_output_dir=false \
#     template=${template}


# #* =============== Merge Lora adapter ===============
# llamafactory-cli export examples/merge_lora/mistral_lora_sft.chatqa.yaml \
#     adapter_name_or_path=output_factory/ChatQA-tmpl-${template}-SFT-lora-Mistral-7B-v0.1/checkpoint-390 \
#     export_dir=output_merged/ChatQA-tmpl-${template}-SFT-lora-Mistral-7B-v0.1 \
#     template=${template}


# #* ==================== Stage 2 =====================
# # export CUDA_VISIBLE_DEVICES=0
# export TRITON_CACHE_DIR='/tmp/triton_cache'
# train_type='lora'
# model_name='/gfshome/LLM-cache/Mistral-7B-v0.1'
# suffix="${model_name##*/}"
# time=$(date +%m%d-%H%M%S)
# llamafactory-cli train examples/train_lora/our_mistral_lora_chatqa_stage2_sft.yaml \
#     model_name_or_path=output_merged/ChatQA-tmpl-${template}-SFT-lora-Mistral-7B-v0.1 \
#     output_dir=output_factory/ChatQA-tmpl-${template}-SFT-stage2-${train_type}-${suffix} \
#     overwrite_output_dir=false \
#     template=${template}