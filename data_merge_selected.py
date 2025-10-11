#!/usr/bin/env python3
import argparse, json, re, sys
from pathlib import Path

PATTERN = re.compile(
    r"^interval-\d+-"
    r"(?P<model>llama3|mistral)_(?P<dataset>nqopen|sciq|triviaqa)"
    r"_train_(?P<kind>align|estimator|reward)_llamafactory\.json$"
)

def load_records(p: Path):
    # 兼容 JSON 数组或 JSONL
    txt = p.read_text(encoding="utf-8").lstrip()
    if txt.startswith("["):
        try:
            data = json.loads(txt)
            if isinstance(data, list): return data
            raise ValueError("JSON is not a list")
        except Exception as e:
            raise ValueError(f"{p}: invalid JSON array: {e}")
    else:
        recs = []
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line: continue
            try:
                recs.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"{p}: JSONL parse error at line {i}: {e}")
        return recs

def main():
    ap = argparse.ArgumentParser(description="Merge sampled JSONs by model×kind")
    ap.add_argument("--input_dir", default="data_xn_sample", help="根目录，含各数据集子目录")
    ap.add_argument("--output_dir", default="data_xn_merged", help="输出目录")
    ap.add_argument("--datasets", nargs="*", default=["nqopen","sciq","triviaqa"],
                    help="只合并这些数据集（子目录名），默认全选")
    ap.add_argument("--models", nargs="*", default=["llama3","mistral"],
                    choices=["llama3","mistral"])
    ap.add_argument("--kinds", nargs="*", default=["align","estimator","reward"],
                    choices=["align","estimator","reward"])
    ap.add_argument("--output_prefix", default="", help="输出文件名前缀（可留空）")
    args = ap.parse_args()

    input_root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 收集匹配文件
    buckets = {(m,k): [] for m in args.models for k in args.kinds}
    for ds in args.datasets:
        ds_dir = input_root / ds
        if not ds_dir.is_dir():
            print(f"[WARN] skip missing dataset dir: {ds_dir}", file=sys.stderr)
            continue
        for p in ds_dir.glob("*.json"):
            m = PATTERN.match(p.name)
            if not m: 
                continue
            meta = m.groupdict()
            if meta["dataset"] not in args.datasets: 
                continue
            if meta["model"] not in args.models or meta["kind"] not in args.kinds:
                continue
            buckets[(meta["model"], meta["kind"])].append(p)

    # 合并输出
    for (model, kind), files in buckets.items():
        files = sorted(files)  # 按文件名排序，保证稳定
        merged = []
        for f in files:
            recs = load_records(f)
            merged.extend(recs)
        out_name = f"{args.output_prefix}{model}_train_{kind}_llamafactory.json"
        out_path = out_root / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fo:
            json.dump(merged, fo, ensure_ascii=False, indent=2)
        print(f"[OK] {model}/{kind}: {len(merged)} items -> {out_path} (from {len(files)} files)")

if __name__ == "__main__":
    main()


# https://chatgpt.com/c/6899eaf5-1028-832d-b3dd-d07dc283dc82
# # 全量合并三个数据集到 6 个文件:
# python data_merge_selected.py --input_dir data_xn_sample --output_dir data_xn_merged
# # 只合并 nqopen 和 sciq 两个数据集、且仅 llama3:
# python data_merge_selected.py --datasets nqopen sciq --models llama3 --output_dir data_xn_merged_llama3
# # 给输出文件加前缀（例如保留“interval-混合”标记）:
# python data_merge_selected.py --output_prefix interval-mixed- --output_dir data_xn_merged

# #* 全量合并三个数据集到 6 个文件:
# python data_merge_selected.py --input_dir data_xn_sample --output_dir data_xn_sample --models llama3 
# python data_merge_selected.py --input_dir data_xn_sample --output_dir data_xn_sample --models mistral

