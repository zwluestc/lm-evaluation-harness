import json
import glob
import os
import re

def main():
    # 找到最新生成的 samples jsonl 文件
    sample_files = glob.glob("results_mixed_pass2/samples_mixed_pass2_*.jsonl")
    if not sample_files:
        sample_files = glob.glob("samples_mixed_pass2_*.jsonl")
        
    if not sample_files:
        print("未找到对应的 samples jsonl 日志文件，请确认评测是否成功生成了结果。")
        return

    # 获取最新的文件
    latest_sample_file = max(sample_files, key=os.path.getctime)
    print(f"解析样本文件: {latest_sample_file}")

    pass_counts = {}
    total_counts = {}
    model_answers = {}
    
    with open(latest_sample_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            doc_id = data.get("doc_id")
            
            is_correct = 0
            if "exact_match" in data:
                is_correct = int(bool(data["exact_match"]))
            elif "metrics" in data and "exact_match" in data["metrics"]:
                is_correct = int(bool(data["metrics"]["exact_match"]))
                
            if doc_id not in pass_counts:
                pass_counts[doc_id] = 0
                total_counts[doc_id] = 0
                model_answers[doc_id] = []
            
            pass_counts[doc_id] += is_correct
            total_counts[doc_id] += 1
            
            # 收集 resps
            resps = data.get("resps", [])
            filtered_resps = data.get("filtered_resps", [])
            # resps 可能是一个列表的列表（特别是 requests 被聚合时）
            if isinstance(resps, list):
                for resp in resps:
                    if isinstance(resp, list):
                        model_answers[doc_id].extend(resp)
                    else:
                        model_answers[doc_id].append(resp)
            else:
                model_answers[doc_id].append(resps)
            
    original_data_path = "data/mixed.jsonl"
    output_data_path = "data/mixed_pass2.jsonl"
    
    if not os.path.exists(original_data_path):
        print(f"找不到原始文件: {original_data_path}")
        return

    print("开始向原数据注入 pass@2 和 qaanswer 并保存到新文件...")
    with open(original_data_path, 'r', encoding='utf-8') as fin, \
         open(output_data_path, 'w', encoding='utf-8') as fout:
        
        for idx, line in enumerate(fin):
            if not line.strip(): continue
            item = json.loads(line)
            
            passes = pass_counts.get(idx, 0)
            total = total_counts.get(idx, 2)
            
            denominator = max(total, 2)
            item["pass@2"] = f"{passes}/{denominator}"
            
            # --- 提取模型推理的最终答案 (提取 <final answer>...</final answer> 中间的内容) ---
            answers_for_doc = model_answers.get(idx, [])
            for i, raw_ans in enumerate(answers_for_doc):
                match = re.search(r'<final answer>(.*?)</final answer>', raw_ans, flags=re.DOTALL)
                if match:
                    extracted_ans = match.group(1).strip()
                else:
                    # 如果模型没有严格输出 <final answer> 标签，则保留它的完整输出
                    extracted_ans = raw_ans.strip()
                item[f"answer{i+1}"] = extracted_ans

            # --- 提取原始题目的答案：在“最终答案”后面 ---
            output_str = item.get("output", "")
            qaanswer = ""
            
            if "最终答案" in output_str:
                # 按照 "最终答案" 切分，取最后一部分
                ans_part = output_str.split("最终答案")[-1].strip()
                if ans_part.startswith("：") or ans_part.startswith(":"):
                    ans_part = ans_part[1:].strip()
                
                # 尝试用测评脚本中相同的 regex (\boxed) 提取具体内容
                matches = re.findall(r'\\boxed\{([^}]*)\}', ans_part)
                if matches:
                    qaanswer = matches[0] # 取提取到的第一个或者唯一的 boxed
                else:
                    qaanswer = ans_part
            else:
                # fallback: 直接在全文找
                matches = re.findall(r'\\boxed\{([^}]*)\}', output_str)
                if matches:
                    qaanswer = matches[-1]
            
            item["qaanswer"] = qaanswer
            
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"处理完毕！结果已保存至 {output_data_path}")

if __name__ == "__main__":
    main()