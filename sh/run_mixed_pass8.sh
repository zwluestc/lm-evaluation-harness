#!/bin/bash

# 1. 设置 HF 国内镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 2. 修正 DSW 中的路径 (根据你报错显示的真实路径)
# 之前的 cd /Users/... 报错是因为那是你本地电脑的路径
PROJECT_DIR="/mnt/data/zwl/verl/lm-evaluation-harness"
cd "$PROJECT_DIR"

# 确保 Python 能找到 lm_eval 模块
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

# 模型路径
MODEL_PATH="/mnt/data/zwl/models/Qwen3-4B-Base"

echo "=================================================="
echo "🚀 使用 vLLM 后端 (8卡并行)，启动评测..."
echo "=================================================="

# 3. 运行评测
# 注意：--tasks mixed_pass8 报错说明你的 sh/ 目录下可能没识别到任务
# 我们显式通过 --include_path 指定任务配置文件所在的目录
python3 -m lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,dtype=bfloat16,tensor_parallel_size=8,gpu_memory_utilization=0.8,trust_remote_code=True \
    --include_path ./sh \
    --tasks mixed_pass8 \
    --apply_chat_template \
    --num_fewshot 0 \
    --batch_size auto \
    --log_samples \
    --output_path ./results_mixed_pass8.json \
    --verbosity INFO 2>&1 | tee eval_pass8.log

echo "=================================================="
echo "✅ 评测完成，现在处理生成结果..."
echo "=================================================="

# 4. 修正后置处理脚本路径
# 根据报错，脚本可能在 scripts/ 下而不是 sh/scripts/ 下
if [ -f "scripts/compute_pass8.py" ]; then
    python3 scripts/compute_pass8.py
else
    echo "⚠️ 找不到 scripts/compute_pass8.py，请确认该脚本的准确位置"
fi