import json

# def check_answer_in_outputs(answer, outputs):
#     """检查 answer 是否出现在 outputs 中"""
#     answer_lower = answer.lower().strip()
#     for output in outputs:
#         if answer_lower in output['greedy_decoding'].lower().strip():
#             return True
#     return False
def check_answer_in_outputs(answer, outputs):
    """检查 answer 是否与 outputs 中的任意一个 greedy_decoding 完全匹配"""
    answer_lower = answer.lower().strip()
    for output in outputs:
        output_text = output['greedy_decoding'].lower().strip()
        if answer_lower == output_text:
            return True
    return False

#* data for SFT estimator
def process_json(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = []
    for item in data:
        question_id = item.get("question_id")
        question = item.get("question")
        answer = item.get("answer")
        confidence = item.get("confidence", 0.0)
        entropy = item.get("entropy", 0.0)
        outputs = item.get("outputs", [])

        if not outputs:
            judgment = "not have knowledge and not honesty"
        else:
            has_answer = check_answer_in_outputs(answer, outputs)
            if has_answer :
                judgment = "have knowledge and honesty"
            elif confidence > 0 :
                judgment = "have knowledge but not honesty"
            elif entropy == 0 and confidence == 0:
                judgment = "not have knowledge and honesty"
            else :
                judgment = "not have knowledge and not honesty"

        

        result.append({
            "question_id": question_id,
            "question": question,
            "answer": judgment
        })

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


#* data for PPO policy model
def process_json_ppo(file1_path, file2_path, output_path):
    # 加载两个 JSON 文件
    with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # 创建一个字典用于快速查找第二个文件的答案
    answer_dict = {item['question_id']: item['answer'] for item in data2}

    # 构造新数据
    new_data = []
    for item in data1:
        qid = item['question_id']
        original_question = item['question']
        original_answer = item['answer']
        updated_answer = answer_dict.get(qid, "")  # 如果没有匹配的，就空字符串

        new_item = {
            "question_id": qid,
            "question": f"{original_question}\n### Self-Eval ###: {original_answer}",
            "answer": updated_answer
        }
        new_data.append(new_item)

    # 保存输出文件
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(new_data, out_f, indent=4, ensure_ascii=False)



#* data for SFT reward model
def map_answer_to_score(answer_text):
    mapping = {
        "have knowledge and honesty": 3,
        "have knowledge but not honesty": 2,
        "not have knowledge and honesty": 1,
        "not have knowledge and not honesty": 0
    }
    return mapping.get(answer_text.strip(), -1)  # 如果找不到映射，返回 -1

def process_and_map_answers(file1_path, file2_path, output_path):
    # 加载两个 JSON 文件
    with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # 创建一个字典用于查找第二个文件中的答案
    answer_dict = {item['question_id']: item['answer'] for item in data2}

    # 构建新数据
    new_data = []
    for item in data1:
        qid = item['question_id']
        original_question = item['question']
        self_eval_answer = item['answer']
        gold_answer = answer_dict.get(qid, "")

        mapped_score = map_answer_to_score(self_eval_answer)

        new_item = {
            "question_id": qid,
            "question": f"{original_question}\n### Self-Eval ###: {self_eval_answer}\n### Gold Answer ###: {gold_answer}",
            "answer": mapped_score
        }
        new_data.append(new_item)

    # 保存输出文件
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(new_data, out_f, indent=4, ensure_ascii=False)


#* convert to llamafactory format
def convert_to_llamafactory_format(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 修改键名
    converted = [
        {"instruction": item["question"], "input": "", "output": str(item["answer"])}
        for item in data
    ]
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=4)

def convert_to_llamafactory_format_rw(input_path, output_path):
    # ===== 映射规则 =====
    output_to_chosen = {
        "0": "-2",
        "1": "-1",
        "2": "1",
        "3": "2"
    }

    # ===== 读取原始数据 =====
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted_data = []

    for item in data:
        # instruction = item.get("instruction", "").strip()
        # output_label = item.get("output", "").strip()
        instruction = item.get("question", "").strip()
        output_label = str(item.get("answer", ""))

        # 映射 output 到 chosen 和 rejected
        chosen_value = output_to_chosen.get(output_label, "0")
        rejected_value = str(-int(chosen_value))

        # 构造新结构
        new_item = {
            "conversations": [
                {
                    "from": "human",
                    "value": instruction
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": chosen_value
            },
            "rejected": {
                "from": "gpt",
                "value": rejected_value
            }
        }

        converted_data.append(new_item)

    # ===== 写入输出文件 =====
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)
    print(f"转换完成，共处理 {len(converted_data)} 条，保存为 {output_path}")



# 示例用法
if __name__ == "__main__":
    # 输入和输出文件路径
    import argparse
    parser = argparse.ArgumentParser(description="Process a JSON file and write to output.")
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to input JSON file")
    parser.add_argument("-o1", "--output_1", required=True, type=str, help="Path to output JSON file") # estimator
    parser.add_argument("-o2", "--output_2", required=True, type=str, help="Path to output JSON file") # ppo
    parser.add_argument("-o3", "--output_3", required=True, type=str, help="Path to output JSON file") # reward
    parser.add_argument("-r", "--raw", required=True, type=str, help="Path to the raw JSON file for PPO and Reward processing")
    args = parser.parse_args()

    if args.output_1 is not None:
        process_json(args.input, args.output_1)
    if args.output_2 is not None:
        process_json_ppo(args.output_1, args.raw, args.output_2)
    if args.output_3 is not None:
        process_and_map_answers(args.output_1, args.raw, args.output_3)

    convert_to_llamafactory_format(args.output_1, args.output_1.replace("estimator", "estimator_llamafactory"))  # Convert to SFT format
    convert_to_llamafactory_format(args.output_2, args.output_2.replace("align", "align_llamafactory"))  # Convert to SFT/PPO format
    convert_to_llamafactory_format_rw(args.output_3, args.output_3.replace("reward", "reward_llamafactory"))  # Convert to RW format
    
    
"""
python data_prompt_construct.py -i data/sciq/prep/llama3_sciq_no_lora_icl/llama3_sciq_train.json -o1 data_xn/sciq/llama3_sciq_train_estimator.json -r data/sciq/raw/train.json -o2 data_xn/sciq/llama3_sciq_train_align.json -o3 data_xn/sciq/llama3_sciq_train_reward.json
python data_prompt_construct.py -i data/nqopen/prep/llama3_nqopen_no_lora_icl/llama3_nqopen_train.json -o1 data_xn/nqopen/llama3_nqopen_train_estimator.json -r data/nqopen/raw/train.json -o2 data_xn/nqopen/llama3_nqopen_train_align.json -o3 data_xn/nqopen/llama3_nqopen_train_reward.json
python data_prompt_construct.py -i data/triviaqa/prep/llama3_triviaqa_no_lora_icl/llama3_triviaqa_train.json -o1 data_xn/triviaqa/llama3_triviaqa_train_estimator.json -r data/triviaqa/raw/train.json -o2 data_xn/triviaqa/llama3_triviaqa_train_align.json  -o3 data_xn/triviaqa/llama3_triviaqa_train_reward.json

python data_prompt_construct.py -i data/sciq/prep/mistral_sciq_no_lora_icl/mistral_sciq_train.json -o1 data_xn/sciq/mistral_sciq_train_estimator.json -r data/sciq/raw/train.json -o2 data_xn/sciq/mistral_sciq_train_align.json -o3 data_xn/sciq/mistral_sciq_train_reward.json
python data_prompt_construct.py -i data/nqopen/prep/mistral_nqopen_no_lora_icl/mistral_nqopen_train.json -o1 data_xn/nqopen/mistral_nqopen_train_estimator.json -r data/nqopen/raw/train.json -o2 data_xn/nqopen/mistral_nqopen_train_align.json -o3 data_xn/nqopen/mistral_nqopen_train_reward.json
python data_prompt_construct.py -i data/triviaqa/prep/mistral_triviaqa_no_lora_icl/mistral_triviaqa_train.json -o1 data_xn/triviaqa/mistral_triviaqa_train_estimator.json -r data/triviaqa/raw/train.json -o2 data_xn/triviaqa/mistral_triviaqa_train_align.json -o3 data_xn/triviaqa/mistral_triviaqa_train_reward.json
"""





        
"""Standalone Usage:
    import argparse
    parser = argparse.ArgumentParser(description="Process a JSON file and write to output.")
    parser.add_argument("-i", "--input", default=None, type=str, help="Path to input JSON file")
    parser.add_argument("-o1", "--output_1", default=None, type=str, help="Path to output JSON file") # estimator
    parser.add_argument("-o2", "--output_2", default=None, type=str, help="Path to output JSON file") # ppo
    parser.add_argument("-o3", "--output_3", default=None, type=str, help="Path to output JSON file") # reward
    parser.add_argument("-r", "--raw", default=None, type=str, help="Path to the raw JSON file for PPO and Reward processing")
    args = parser.parse_args()
    
    if args.output_1 is not None:
        process_json(args.input, args.output_1)
    if args.output_2 is not None:
        process_json_ppo(args.input, args.raw, args.output_2)
    if args.output_3 is not None:
        process_and_map_answers(args.input, args.raw, args.output_3)
#------estimator
python data_prompt_construct.py -i data/sciq/prep/llama3_sciq_no_lora_icl/llama3_sciq_train.json -o1 data_xn/sciq/llama3_sciq_train_estimator.json
python data_prompt_construct.py -i data/nqopen/prep/llama3_nqopen_no_lora_icl/llama3_nqopen_train.json -o1 data_xn/nqopen/llama3_nqopen_train_estimator.json
python data_prompt_construct.py -i data/triviaqa/prep/llama3_triviaqa_no_lora_icl/llama3_triviaqa_train.json -o1 data_xn/triviaqa/llama3_triviaqa_train_estimator.json

python data_prompt_construct.py -i data/sciq/prep/mistral_sciq_no_lora_icl/mistral_sciq_train.json -o1 data_xn/sciq/mistral_sciq_train_estimator.json
python data_prompt_construct.py -i data/nqopen/prep/mistral_nqopen_no_lora_icl/mistral_nqopen_train.json -o1 data_xn/nqopen/mistral_nqopen_train_estimator.json
python data_prompt_construct.py -i data/triviaqa/prep/mistral_triviaqa_no_lora_icl/mistral_triviaqa_train.json -o1 data_xn/triviaqa/mistral_triviaqa_train_estimator.json

#------PPO
python data_prompt_construct.py -i data_xn/sciq/llama3_sciq_train_estimator.json -r data/sciq/raw/train.json -o2 data_xn/sciq/llama3_sciq_train_align.json
python data_prompt_construct.py -i data_xn/nqopen/llama3_nqopen_train_estimator.json -r data/nqopen/raw/train.json -o2 data_xn/nqopen/llama3_nqopen_train_align.json
python data_prompt_construct.py -i data_xn/triviaqa/llama3_triviaqa_train_estimator.json -r data/triviaqa/raw/train.json -o2 data_xn/triviaqa/llama3_triviaqa_train_align.json

python data_prompt_construct.py -i data_xn/sciq/mistral_sciq_train_estimator.json -r data/sciq/raw/train.json -o2 data_xn/sciq/mistral_sciq_train_align.json
python data_prompt_construct.py -i data_xn/nqopen/mistral_nqopen_train_estimator.json -r data/nqopen/raw/train.json -o2 data_xn/nqopen/mistral_nqopen_train_align.json
python data_prompt_construct.py -i data_xn/triviaqa/mistral_triviaqa_train_estimator.json -r data/triviaqa/raw/train.json -o2 data_xn/triviaqa/mistral_triviaqa_train_align.json

#------Reward
python data_prompt_construct.py -i data_xn/sciq/llama3_sciq_train_estimator.json -r data/sciq/raw/train.json -o3 data_xn/sciq/llama3_sciq_train_reward.json
python data_prompt_construct.py -i data_xn/nqopen/llama3_nqopen_train_estimator.json -r data/nqopen/raw/train.json -o3 data_xn/nqopen/llama3_nqopen_train_reward.json
python data_prompt_construct.py -i data_xn/triviaqa/llama3_triviaqa_train_estimator.json -r data/triviaqa/raw/train.json -o3 data_xn/triviaqa/llama3_triviaqa_train_reward.json

python data_prompt_construct.py -i data_xn/sciq/mistral_sciq_train_estimator.json -r data/sciq/raw/train.json -o3 data_xn/sciq/mistral_sciq_train_reward.json
python data_prompt_construct.py -i data_xn/nqopen/mistral_nqopen_train_estimator.json -r data/nqopen/raw/train.json -o3 data_xn/nqopen/mistral_nqopen_train_reward.json
python data_prompt_construct.py -i data_xn/triviaqa/mistral_triviaqa_train_estimator.json -r data/triviaqa/raw/train.json -o3 data_xn/triviaqa/mistral_triviaqa_train_reward.json


"""