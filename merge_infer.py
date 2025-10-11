import json
import argparse

def merge_json_and_jsonl(json_path, jsonl_path, output_path):
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 读取 JSONL 文件
    jsonl_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            jsonl_data.append(json.loads(line.strip()))

    # 合并数据
    merged_data = []
    for idx, (json_item, jsonl_item) in enumerate(zip(json_data, jsonl_data)):
        merged_item = {
            "question_id": json_item.get("question_id", f"id_{idx+1}"),
            "question": json_item.get("question"),
            "answer": json_item.get("answer"),
            "output": jsonl_item.get("predict", "")
        }
        merged_data.append(merged_item)

    # 写入输出 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f"Merged {len(merged_data)} items and saved to {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a JSON file and a JSONL file into one JSON output.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged JSON output.")
    args = parser.parse_args()
    
    merge_json_and_jsonl(json_path=args.json_path, jsonl_path=args.jsonl_path, output_path=args.output_path)
    