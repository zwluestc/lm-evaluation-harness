import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor

# 提示：请根据你的实际使用的 LLM API 安装对应的库，例如 openai
# pip install openai
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

def query_llm_as_judge(correct_answer, model_answer, client, model_name):
    """
    使用 LLM 作为裁判，判断模型给出的答案是否等价于标准答案。
    """
    if not model_answer:
        return "No (Empty)"

    prompt = f"""你是一个严格且专业的评分系统的裁判专家。你需要判断【模型预测答案】是否在数学、逻辑或语义上等价于【标准答案】。

【标准答案】
{correct_answer}

【模型预测答案】
{model_answer}

判断要求：
1. 只要模型预测答案的最终结果等价于标准答案，就认为正确。
2. 忽略格式差异（比如多余的空格、LaTeX括号等）。
3. 如果预测答案完全错误或未给出有效结论，则为错误。

请给出你的判定结果。只需输出 "Yes"（完全正确等价）或者 "No"（错误）。不要输出任何其他解释。"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        result = response.choices[0].message.content.strip()
        if "Yes" in result or "yes" in result or "YES" in result:
            return "Yes"
        else:
            return "No"
    except Exception as e:
        print(f"API Error: {e}")
        return "Error"

def process_item(item, client, model_name):
    qaanswer = item.get("qaanswer", "")
    
    # 找到所有的 answer1, answer2 ...
    answer_keys = [k for k in item.keys() if k.startswith("answer") and k != "qaanswer"]
    
    for ans_key in answer_keys:
        judge_key = f"judge_{ans_key}"
        # 如果已经判断过了，可以跳过
        if judge_key in item and item[judge_key] not in ["Error", ""]:
            continue
            
        model_answer = item.get(ans_key, "")
        if qaanswer and model_answer:
            judge_res = query_llm_as_judge(qaanswer, model_answer, client, model_name)
            item[judge_key] = judge_res
        else:
            item[judge_key] = "No (Missing Data)"
            
    return item

def main():
    parser = argparse.ArgumentParser(description="使用 LLM 判题裁判")
    parser.add_argument("--input", type=str, default="data/mixed_pass2.jsonl", help="输入/输出的 jsonl 文件")
    parser.add_argument("--model", type=str, default="gpt-4o", help="判题 LLM 模型名称")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="API Key")
    parser.add_argument("--base_url", type=str, default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="API Base URL")
    parser.add_argument("--workers", type=int, default=4, help="并发请求的数量")
    args = parser.parse_args()

    if OpenAI is None:
        print("请先安装 openai 库: pip install openai")
        return

    if not args.api_key:
        print("请提供环境变量 OPENAI_API_KEY 或者通过 --api_key 传入。")
        return

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    if not os.path.exists(args.input):
        print(f"找不到文件: {args.input}")
        return

    # 加载数据
    data = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data.append(json.loads(line))

    print(f"总计加载了 {len(data)} 条数据，准备进行 LLM 判题 (模型: {args.model})...")

    # 并发处理
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for item in data:
            futures.append(executor.submit(process_item, item, client, args.model))
            
        for future in futures:
            results.append(future.result())

    # 写回文件前，根据 judge 结果重新计算 pass@2
    for item in results:
        correct_count = 0
        total_count = 0
        # 遍历所有的判断结果字段
        judge_keys = [k for k in item.keys() if k.startswith("judge_answer")]
        for jk in judge_keys:
            total_count += 1
            if item[jk] == "Yes":
                correct_count += 1
        
        denominator = max(total_count, 2)
        item["pass@2"] = f"{correct_count}/{denominator}"

    # 写回文件
    with open(args.input, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"判题完成，所有的 judge 结果已经回写保存到了 {args.input}")

if __name__ == "__main__":
    main()