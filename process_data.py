import os
import json
import pandas as pd
import argparse
from pathlib import Path

# process raw nqopen data from parquet to jsonl format
def process_parquet_format():
    # create nqopen dir, git clone the dataset, only keep the nq_open folder and mv to nqopen directory.
    os.makedirs("data/nqopen", exist_ok=True)

    # read parquet file
    df = pd.read_parquet("/gfshome/UAlign/data/nqopen/nq_open/train-00000-of-00001.parquet")

    # save as JSONL file
    df.to_json("/gfshome/UAlign/data/nqopen/raw/NQ-open.train.jsonl", 
            orient="records", 
            lines=True, 
            force_ascii=False)

    df = pd.read_parquet("/gfshome/UAlign/data/nqopen/nq_open/validation-00000-of-00001.parquet")

    # save as JSONL file
    df.to_json("/gfshome/UAlign/data/nqopen/raw/NQ-open.validation.jsonl", 
            orient="records", 
            lines=True, 
            force_ascii=False)


# raw jsonl to raw json is omited
...


# 从raw json再改成alpaca格式: keyname: question -> instruction, answer -> output, input -> ""
def convert_format(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = []
    for item in data:
        new_item = {
            "instruction": item.get("question", ""),
            "input": "",
            "output": item.get("answer", "")
        }
        new_data.append(new_item)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)


# 用estimator推理raw alpaca,生成save/xxx.jsonl,将其合并到data_xn-validation
def process_line(line):
    obj = json.loads(line)
    # 提取内容
    prompt = obj["prompt"]
    predict = obj["predict"]
    label = obj["label"]

    # 提取真实指令（去掉 "Human:" 和 "</s>" 之后的内容）
    if prompt.startswith("Human:"):
        instruction_part = prompt[len("Human:"):].split("</s>")[0].strip()
    else:
        instruction_part = prompt.strip()

    # 清洗换行符和空格
    instruction = f"{instruction_part}\n### Self-Eval ###: {predict.strip()}"
    output = label.strip()

    return {
        "instruction": instruction,
        "input": "",
        "output": output
    }

def extract_from_llamafactory(input_path, output_path): # 对在LLaMA-Factory中对Estimator进行推理得到的jsonl文件进行处理
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    os.makedirs(output_path.parent, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as infile:
        lines = infile.readlines()

    results = [process_line(line) for line in lines]

    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(results, outfile, indent=2, ensure_ascii=False)




if __name__ == "__main__":
    
    # process_parquet_format() # Finished; Uncomment to process parquet files
    
    parser = argparse.ArgumentParser(description="Convert QA JSON format to instruction format.")
    parser.add_argument('--input', type=str, default=None, help='Path to input JSON file')
    parser.add_argument('--output', type=str, default=None, help='Path to output JSON file')

    parser.add_argument("--input_path", default=None,  help="Path to the input .jsonl file")
    parser.add_argument("--output_path", default=None,  help="Path to the output .json file")

    args = parser.parse_args()
    if args.input is not None and args.output is not None: # 非None才执行这个函数
        convert_format(args.input, args.output)
        
    if args.input_path is not None and args.output_path is not None: # 非None才执行这个函数
        extract_from_llamafactory(args.input_path, args.output_path)


# https://chatgpt.com/c/68a2089f-5d24-8333-8857-8910aada65e5
# # Usage examples:
# python process_data.py --input data/nqopen/raw/validation.json --output data/nqopen/raw/validation-alpaca.json
# python process_data.py --input_path LLaMA-Factory/saves/Mistral-7B-v0.1/lora/estimator_inference_mistral_nqopen_val/generated_predictions.jsonl --output_path data_xn-validation/nqopen/mistral_validation_NLI.json