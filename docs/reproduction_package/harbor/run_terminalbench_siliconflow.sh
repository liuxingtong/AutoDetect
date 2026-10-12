#!/bin/bash
#===============================================================================
# TerminalBench 运行脚本 - 硅基流动 (SiliconFlow) API 版
#
# 与 run_terminalbench_improved.sh 的区别：
#   - 不再使用本地 Qwen3.5-9B + transformers serve
#   - 通过 SiliconFlow OpenAI 兼容 API 调用 Qwen/Qwen3.5-35B-A3B
#   - API key 从环境变量 SILICONFLOW_API_KEY 读取（详见 .env.example）
#
# 用法：
#   1. cp .env.example .env && 编辑 .env 填入真实 key
#   2. bash run_terminalbench_siliconflow.sh
#===============================================================================

set -e

# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
REMOTE_HOST="liuxingtong@115.182.62.246"
REMOTE_DIR="/data/liuxingtong"
HARBOR_JOBS_DIR="/data/liuxingtong/harbor/jobs"
BENCHMARK_DIR="/data/liuxingtong/official-benchmarks/terminal-bench-2"
LOCAL_CONFIG="docs/reproduction_package/harbor/terminalbench-qwen35-35b-siliconflow.json"
REMOTE_CONFIG="/data/liuxingtong/harbor/jobs/terminalbench-qwen35-35b-siliconflow.json"

# ----------------------------------------------------------------------------
# 1. 本地：检查 .env 和 API key
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "[!] 未找到 $ENV_FILE"
    echo "    请先: cp $SCRIPT_DIR/.env.example $ENV_FILE"
    echo "    然后编辑 $ENV_FILE 填入 SILICONFLOW_API_KEY"
    exit 1
fi

# shellcheck disable=SC1090
set -a
source "$ENV_FILE"
set +a

if [ -z "${SILICONFLOW_API_KEY:-}" ] || [[ "$SILICONFLOW_API_KEY" == sk-your-* ]]; then
    echo "[!] SILICONFLOW_API_KEY 未设置或仍是占位符"
    echo "    请编辑 $ENV_FILE"
    exit 1
fi
echo "[1] API key 已加载（前缀: ${SILICONFLOW_API_KEY:0:7}...）"

# ----------------------------------------------------------------------------
# 2. 远端：把 config 推上去
# ----------------------------------------------------------------------------
echo "[2] 推送 config 到 $REMOTE_HOST ..."
scp -o StrictHostKeyChecking=no "$SCRIPT_DIR/$LOCAL_CONFIG" \
    "$REMOTE_HOST:$REMOTE_CONFIG"

# ----------------------------------------------------------------------------
# 3. 远端：先做一次最小 API 烟雾测试（不开 docker，最快暴露网络/key 问题）
# ----------------------------------------------------------------------------
echo "[3] 远端 API 烟雾测试 ..."
ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" \
    "SILICONFLOW_API_KEY='$SILICONFLOW_API_KEY' curl -sS --max-time 30 \
        https://api.siliconflow.cn/v1/chat/completions \
        -H 'Authorization: Bearer '$SILICONFLOW_API_KEY \
        -H 'Content-Type: application/json' \
        -d '{
            \"model\": \"Qwen/Qwen3.5-35B-A3B\",
            \"messages\": [{\"role\":\"user\",\"content\":\"say ok in one word\"}],
            \"max_tokens\": 16,
            \"temperature\": 0.0
        }' | head -c 400"

echo ""
echo "[4] 启动 harbor ..."
echo "    harbor run $REMOTE_CONFIG"
echo "    （按 Ctrl+C 取消；如果进入 Docker trial，请勿中断）"
echo ""

ssh -o StrictHostKeyChecking=no -tt "$REMOTE_HOST" \
    "SILICONFLOW_API_KEY='$SILICONFLOW_API_KEY' harbor run '$REMOTE_CONFIG'"
