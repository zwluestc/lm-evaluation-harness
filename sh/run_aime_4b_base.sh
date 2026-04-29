#!/bin/bash

# 设置 HF 国内镜像源
export HF_ENDPOINT=https://hf-mirror.com

cd /mnt/data/zwl/lm-evaluation-harness
export PYTHONPATH=$PYTHONPATH:$(pwd)

MODEL_PATH="/mnt/data/zwl/models/Qwen3-4B-Base"

echo "=================================================="
echo "使用原生 HF 后端，启动 2 卡并行评测..."
echo "=================================================="

# 使用 accelerate 启动  卡并行
accelerate launch --num_processes=2 -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=bfloat16 \
    --include_path ./custom_tasks \
    --tasks aime24_base \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --num_fewshot 0 \
    --batch_size auto \
    --log_samples \
    --output_path ./results_aime24_pass1.json \
    --verbosity INFO 2>&1 | tee eval_pass1.log
    