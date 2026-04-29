#!/bin/bash

# 设置 HF 国内镜像源
export HF_ENDPOINT=https://hf-mirror.com

cd /Users/zwlustc/Documents/lm-evaluation-harness
export PYTHONPATH=$PYTHONPATH:$(pwd)

MODEL_PATH="/mnt/data/zwl/models/Qwen3-4B-Base"

echo "=================================================="
echo "使用原生 HF 后端，启动评测 (8次采样以计算 pass@8)..."
echo "=================================================="

# 使用 accelerate 启动 
accelerate launch --num_processes=2 -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=bfloat16 \
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
