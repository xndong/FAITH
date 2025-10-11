#### merge inference results

# Mistral
# python merge_infer.py \
#     --json_path data/nqopen/raw/validation.json \
#     --jsonl_path LLaMA-Factory/saves/Mistral-7B-v0.1/lora/eval_mistral_nqopen/generated_predictions.jsonl \
#     --output_path data_xn-validation-results/mistral_nqopen_validation_results.json

# python merge_infer.py \
#     --json_path data/sciq/raw/validation.json \
#     --jsonl_path LLaMA-Factory/saves/Mistral-7B-v0.1/lora/eval_mistral_sciq/generated_predictions.jsonl \
#     --output_path data_xn-validation-results/mistral_sciq_validation_results.json

# python merge_infer.py \
#     --json_path data/triviaqa/raw/validation.json \
#     --jsonl_path LLaMA-Factory/saves/Mistral-7B-v0.1/lora/eval_mistral_triviaqa/generated_predictions.jsonl \
#     --output_path data_xn-validation-results/mistral_triviaqa_validation_results.json

# python merge_infer.py \
#     --json_path data/simpleqa/raw/validation.json \
#     --jsonl_path LLaMA-Factory/saves/Mistral-7B-v0.1/lora/eval_mistral_simpleqa/generated_predictions.jsonl \
#     --output_path data_xn-validation-results/mistral_simpleqa_validation_results.json

# python merge_infer.py \
#     --json_path data/webquestionqa/raw/validation.json \
#     --jsonl_path LLaMA-Factory/saves/Mistral-7B-v0.1/lora/eval_mistral_webquestionqa/generated_predictions.jsonl \
#     --output_path data_xn-validation-results/mistral_webquestionqa_validation_results.json

#==================== RAG data ====================

# python merge_infer.py \
#     --json_path data/nqopen/raw/validation.json \
#     --jsonl_path LLaMA-Factory/saves/Mistral-7B-v0.1/lora/rag_eval_mistral_nqopen/generated_predictions.jsonl \
#     --output_path data_xn-validation-results-rag/mistral_nqopen_validation_results.json

# python merge_infer.py \
#     --json_path data/sciq/raw/validation.json \
#     --jsonl_path LLaMA-Factory/saves/Mistral-7B-v0.1/lora/rag_eval_mistral_sciq/generated_predictions.jsonl \
#     --output_path data_xn-validation-results-rag/mistral_sciq_validation_results.json

# python merge_infer.py \
#     --json_path data/triviaqa/raw/validation.json \
#     --jsonl_path LLaMA-Factory/saves/Mistral-7B-v0.1/lora/rag_eval_mistral_triviaqa/generated_predictions.jsonl \
#     --output_path data_xn-validation-results-rag/mistral_triviaqa_validation_results.json

# python merge_infer.py \
#     --json_path data/webquestionqa/raw/validation.json \
#     --jsonl_path LLaMA-Factory/saves/Mistral-7B-v0.1/lora/rag_eval_mistral_webquestionqa/generated_predictions.jsonl \
#     --output_path data_xn-validation-results-rag/mistral_webquestionqa_validation_results.json


#==================== 09.10 SFTmixRAG data ====================

# python merge_infer.py \
#     --json_path data/nqopen/raw/validation.json \
#     --jsonl_path LLaMA-Factory/saves/Mistral-7B-v0.1/lora/sftmixrag_eval_mistral_nqopen/generated_predictions.jsonl \
#     --output_path data_xn-validation-results-rag/mistral_nqopen_validation_results.json

# python merge_infer.py \
#     --json_path data/sciq/raw/validation.json \
#     --jsonl_path LLaMA-Factory/saves/Mistral-7B-v0.1/lora/sftmixrag_eval_mistral_sciq/generated_predictions.jsonl \
#     --output_path data_xn-validation-results-rag/mistral_sciq_validation_results.json

# python merge_infer.py \
#     --json_path data/triviaqa/raw/validation.json \
#     --jsonl_path LLaMA-Factory/saves/Mistral-7B-v0.1/lora/sftmixrag_eval_mistral_triviaqa/generated_predictions.jsonl \
#     --output_path data_xn-validation-results-rag/mistral_triviaqa_validation_results.json

# python merge_infer.py \
#     --json_path data/webquestionqa/raw/validation.json \
#     --jsonl_path LLaMA-Factory/saves/Mistral-7B-v0.1/lora/sftmixrag_eval_mistral_webquestionqa/generated_predictions.jsonl \
#     --output_path data_xn-validation-results-rag/mistral_webquestionqa_validation_results.json


#==================== 09.23 Estimator vs. Sample@K ====================
# python merge_infer.py \
#     --json_path data/nqopen/raw/validation.json \
#     --jsonl_path LLaMA-Factory/saves/Mistral-7B-v0.1/lora/sampleK_eval_mistral_nqopen/generated_predictions.jsonl \
#     --output_path data_xn-validation-results-sampleK/mistral_nqopen_validation_results.json

# python merge_infer.py \
#     --json_path data/sciq/raw/validation.json \
#     --jsonl_path LLaMA-Factory/saves/Mistral-7B-v0.1/lora/sampleK_eval_mistral_sciq/generated_predictions.jsonl \
#     --output_path data_xn-validation-results-sampleK/mistral_sciq_validation_results.json

# python merge_infer.py \
#     --json_path data/triviaqa/raw/validation.json \
#     --jsonl_path LLaMA-Factory/saves/Mistral-7B-v0.1/lora/sampleK_eval_mistral_triviaqa/generated_predictions.jsonl \
#     --output_path data_xn-validation-results-sampleK/mistral_triviaqa_validation_results.json

# python merge_infer.py \
#     --json_path data/webquestionqa/raw/validation.json \
#     --jsonl_path LLaMA-Factory/saves/Mistral-7B-v0.1/lora/sampleK_eval_mistral_webquestionqa/generated_predictions.jsonl \
#     --output_path data_xn-validation-results-sampleK/mistral_webquestionqa_validation_results.json


#==================== 09.24 Training Scaling K=12 ====================
python merge_infer.py \
    --json_path data/nqopen/raw/validation.json \
    --jsonl_path LLaMA-Factory/saves/Llama-3-8B/lora/eval_llama3_nqopen/generated_predictions.jsonl \
    --output_path data_xn-validation-results/llama3_nqopen_validation_results.json

python merge_infer.py \
    --json_path data/sciq/raw/validation.json \
    --jsonl_path LLaMA-Factory/saves/Llama-3-8B/lora/eval_llama3_sciq/generated_predictions.jsonl \
    --output_path data_xn-validation-results/llama3_sciq_validation_results.json

python merge_infer.py \
    --json_path data/triviaqa/raw/validation.json \
    --jsonl_path LLaMA-Factory/saves/Llama-3-8B/lora/eval_llama3_triviaqa/generated_predictions.jsonl \
    --output_path data_xn-validation-results/llama3_triviaqa_validation_results.json

python merge_infer.py \
    --json_path data/webquestionqa/raw/validation.json \
    --jsonl_path LLaMA-Factory/saves/Llama-3-8B/lora/eval_llama3_webquestionqa/generated_predictions.jsonl \
    --output_path data_xn-validation-results/llama3_webquestionqa_validation_results.json





# Llama3




### evaluate/score inference results

# Mistral
if [ -z "$1" ]; then
    echo "用法: $0 <ppo/sft_types>"
    exit 1
fi
if [ -z "$2" ]; then
    echo "用法: $0 <ppo/sft_steps>"
    exit 1
fi

# python eval.py --model_name mistral --dataset nqopen --data_file validation --train_type "$1" --score_use --steps "$2"
# python eval.py --model_name mistral --dataset sciq --data_file validation --train_type "$1" --score_use --steps "$2"
# python eval.py --model_name mistral --dataset triviaqa --data_file validation --train_type "$1" --score_use --steps "$2"
# python eval.py --model_name mistral --dataset simpleqa --data_file validation --train_type "$1" --score_use --steps "$2"
# python eval.py --model_name mistral --dataset webquestionqa --data_file validation --train_type "$1" --score_use --steps "$2"


#==================== RAG data ====================

# python eval.py --model_name mistral --dataset nqopen --data_file validation --train_type "$1" --score_use --steps "$2" --input_dir /gfshome/UAlign/data_xn-validation-results-rag
# python eval.py --model_name mistral --dataset sciq --data_file validation --train_type "$1" --score_use --steps "$2" --input_dir /gfshome/UAlign/data_xn-validation-results-rag
# python eval.py --model_name mistral --dataset triviaqa --data_file validation --train_type "$1" --score_use --steps "$2" --input_dir /gfshome/UAlign/data_xn-validation-results-rag
# python eval.py --model_name mistral --dataset webquestionqa --data_file validation --train_type "$1" --score_use --steps "$2" --input_dir /gfshome/UAlign/data_xn-validation-results-rag


#==================== 09.10 SFTmixRAG data ====================

# python eval.py --model_name mistral --dataset nqopen --data_file validation --train_type "$1" --score_use --steps "$2" --input_dir /gfshome/UAlign/data_xn-validation-results-rag
# python eval.py --model_name mistral --dataset sciq --data_file validation --train_type "$1" --score_use --steps "$2" --input_dir /gfshome/UAlign/data_xn-validation-results-rag
# python eval.py --model_name mistral --dataset triviaqa --data_file validation --train_type "$1" --score_use --steps "$2" --input_dir /gfshome/UAlign/data_xn-validation-results-rag
# python eval.py --model_name mistral --dataset webquestionqa --data_file validation --train_type "$1" --score_use --steps "$2" --input_dir /gfshome/UAlign/data_xn-validation-results-rag


#==================== 09.23 Estimator vs. Sample@K ====================
# python eval.py --model_name mistral --dataset nqopen --data_file validation --train_type "$1" --score_use --steps "$2" --input_dir /gfshome/UAlign/data_xn-validation-results-sampleK
# python eval.py --model_name mistral --dataset sciq --data_file validation --train_type "$1" --score_use --steps "$2" --input_dir /gfshome/UAlign/data_xn-validation-results-sampleK
# python eval.py --model_name mistral --dataset triviaqa --data_file validation --train_type "$1" --score_use --steps "$2" --input_dir /gfshome/UAlign/data_xn-validation-results-sampleK
# python eval.py --model_name mistral --dataset webquestionqa --data_file validation --train_type "$1" --score_use --steps "$2" --input_dir /gfshome/UAlign/data_xn-validation-results-sampleK


#==================== 09.24 Training Scaling K=12 ====================
python eval.py --model_name llama3 --dataset nqopen --data_file validation --train_type "$1" --score_use --steps "$2"
python eval.py --model_name llama3 --dataset sciq --data_file validation --train_type "$1" --score_use --steps "$2"
python eval.py --model_name llama3 --dataset triviaqa --data_file validation --train_type "$1" --score_use --steps "$2"
python eval.py --model_name llama3 --dataset webquestionqa --data_file validation --train_type "$1" --score_use --steps "$2"



# Llama3
