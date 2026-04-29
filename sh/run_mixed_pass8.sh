#!/bin/bash

# 设置 HF 国内镜像源
export HF_ENDPOINT=https://hf-mirror.com

cd /Users/zwlustc/Documents/lm-evaluation-harness
export PYTHONPATH=$PYTHONPATH:$(pwd)

MODEL_PATH="/mnt/data/zwl/models/Qwen3-4B-Base"

echo "=================================================="
echo "使用 vLLM 后端，启动评测 (8次采样以计算 pass@8)..."
echo "实时日志保存在 eval_pass8.log"
echo "=================================================="

# 使用 vllm 后端
python -m lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,dtype=bfloat16,tensor_parallel_size=8,gpu_memory_utilization=0.8,add_bos_token=True \
    --include_path ./sh \
    --tasks mixed_pass8 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --num_fewshot 0 \
    --batch_size auto \
    --log_samples \
    --output_path ./results_mixed_pass8.json \
    --verbosity INFO 2>&1 | tee eval_pass8.log

echo "=================================================="
echo "评测完成，现在处理生成结果并注入 pass@8 指标到 jsonl 中..."
echo "=================================================="

# 运行后置处理脚本计算每个样本的 pass@8 并写回/生成新的 jsonl
python scripts/compute_pass8.py
