import json
import glob
import os

def main():
    # 找到最新生成的 samples jsonl 文件
    sample_files = glob.glob("samples_mixed_pass8_*.jsonl")
    if not sample_files:
        # 有可能是在 output_path 对应的目录里
        sample_files = glob.glob("results_mixed_pass8.json/*samples_mixed_pass8_*.jsonl")
        
    if not sample_files:
        print("未找到对应的 samples jsonl 日志文件，请确认 --log_samples 是否生效或评测是否成功结束。")
        return

    # 获取最新的文件
    latest_sample_file = max(sample_files, key=os.path.getctime)
    print(f"解析样本文件: {latest_sample_file}")

    # 读取评测结果，通常记录了 doc_id, exact_match, resps 等
    # lm-evaluation-harness 的 repeats 运行后，如果使用 metrics (如 exact_match)
    # 对于每个 doc_id，会生成多条记录（如果按照 doc_id 分开记录的话）
    # 在有些版本中，重复采样会作为一个 list 记录在 target 或 resps 中
    
    pass_counts = {}
    total_counts = {}
    
    with open(latest_sample_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            doc_id = data.get("doc_id")
            
            # 评测标准：若使用了 filter_list 提取 exact_match，结果位于 exact_match 字段
            # True / False 或者是具体的得分 1 / 0
            is_correct = 0
            # compatibility width different lm-eval versions
            if "exact_match" in data:
                is_correct = int(bool(data["exact_match"]))
            elif "metrics" in data and "exact_match" in data["metrics"]:
                is_correct = int(bool(data["metrics"]["exact_match"]))
                
            if doc_id not in pass_counts:
                pass_counts[doc_id] = 0
                total_counts[doc_id] = 0
            
            # 这里累加单个通过的次数
            pass_counts[doc_id] += is_correct
            total_counts[doc_id] += 1
            
    # 如果 lm-eval 版本将所有 resps 聚合在一条数据里，resps 是一个列表
    # 则需要从 resps 判断 （在比较新的 lm-eval-harness 结合 repeats 时，也会这样组织）
    # 由于不确定具体版本，我们双重检查：如果只读到一次记录，但 resps 有八个结果的情况
    
    original_data_path = "data/mixed.jsonl"
    output_data_path = "data/mixed_pass8.jsonl"
    
    if not os.path.exists(original_data_path):
        print(f"找不到原始文件: {original_data_path}")
        return

    print("开始向原数据注入 pass@8 并保存到新文件...")
    with open(original_data_path, 'r', encoding='utf-8') as fin, \
         open(output_data_path, 'w', encoding='utf-8') as fout:
        
        for idx, line in enumerate(fin):
            if not line.strip(): continue
            item = json.loads(line)
            
            # 使用对应 doc_id 的测试结果，如果没有就默认为 0/8
            passes = pass_counts.get(idx, 0)
            total = total_counts.get(idx, 8)
            
            # 如果某些样本收集的不是8次（比如因为某些原因），统一下分母
            if total == 1 and total_counts.get(idx, 0) == 1:
                # 可能是聚合到一条记录里了，没有展开
                pass
                
            denominator = max(total, 8)
            item["pass@8"] = f"{passes}/{denominator}"
            
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"处理完毕！结果已保存至 {output_data_path}")

if __name__ == "__main__":
    main()
