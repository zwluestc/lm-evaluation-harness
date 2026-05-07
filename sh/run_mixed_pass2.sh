#!/bin/bash

# 设置您的 LLM 裁判配置 (请在这里填入真实的 API Key)
export OPENAI_API_KEY="sk-xxxxxxxxxxx"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export JUDGE_MODEL="gpt-4o"

# 设置 HF 国内镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 切换到项目根目录
WORK_DIR=$(cd "$(dirname "$0")/.."; pwd)
cd "$WORK_DIR"
export PYTHONPATH=$PYTHONPATH:$WORK_DIR

# 模型路径（请根据实际情况修改）
MODEL_PATH="/mnt/data/zwl/models/Qwen3-8B"

echo "=================================================="
echo "使用 vLLM 后端，启动 pass@2 评测..."
echo "模型: $MODEL_PATH"
echo "任务: mixed_pass2"
echo "日志保存在 eval_vllm_mixed_pass2.log"
echo "=================================================="

# 运行 lm_eval 评测
python -m lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,dtype=bfloat16,tensor_parallel_size=8,gpu_memory_utilization=0.8,add_bos_token=True \
    --include_path ./sh \
    --tasks mixed_pass2 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --num_fewshot 0 \
    --batch_size auto \
    --log_samples \
    --output_path ./results_mixed_pass2 \
    --verbosity INFO 2>&1 | tee eval_vllm_mixed_pass2.log

echo "=================================================="
echo "生成阶段完成！"
echo "总体指标和逐条样本结果已保存在 ./results_mixed_pass2 目录下"
echo "=================================================="

echo "=================================================="
echo "步骤 2: 提取模型推理的答案并合并到临时文件..."
python scripts/compute_pass2.py

echo "=================================================="
echo "步骤 3: 启动 LLM 作为裁判，对推理答案进行最终等价性判定..."
python scripts/llm_as_judge.py \
    --input data/mixed_pass2.jsonl \
    --model "$JUDGE_MODEL" \
    --workers 4

echo "=================================================="
echo "全流程完成！二次校验和提取得到的答复都已保存并更新了 pass@2"
echo "请查看: data/mixed_pass2.jsonl"
echo "=================================================="