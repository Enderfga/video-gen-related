#!/bin/bash

# ================= 配置区域 =================
BASE_DIR="/root/data/video-gen-related/outputs/vbench"
TARGET_DIR="fastvideo_i2v"
GOAL_TOTAL=5590
GOAL_PER_MACHINE=2795

REMOTE_HOST="root@31.22.104.21"
REMOTE_BASE_DIR="/root/data/video-gen-related/outputs/vbench"
VBENCH_JSON="/root/data/FAR-World/assets/data/meta/vbench/vbench2_i2v_aug_full_info.json"

REFRESH_RATE=5
# ===========================================

mkdir -p "$BASE_DIR/$TARGET_DIR"

# 生成快速统计脚本
STAT_SCRIPT="/tmp/vbench_stat.py"
cat > "$STAT_SCRIPT" << 'PYEOF'
import json, os, sys

json_path = sys.argv[1]
output_dir = sys.argv[2]

with open(json_path) as f:
    data = json.load(f)

mid = len(data) // 2
m0_prompts = set(item["prompt_en"] for item in data[:mid])

m0_done, m1_done = 0, 0

if os.path.exists(output_dir):
    for f in os.listdir(output_dir):
        if not f.endswith('.mp4'):
            continue
        name = f.replace('.mp4', '')
        prompt = name.rsplit('-', 1)[0] if name[-1:].isdigit() and len(name) > 2 and name[-2] == '-' else name

        if prompt in m0_prompts:
            m0_done += 1
        else:
            m1_done += 1

print(m0_done, m1_done)
PYEOF

START_TIME=$(date +%s)
FIRST_RUN=true
TOTAL_START_COUNT=0

format_time() {
    local T=$1
    printf "%02d:%02d:%02d" $((T/3600)) $(((T%3600)/60)) $((T%60))
}

while true; do
    CURRENT_TIME=$(date +%s)

    # 本地统计
    read LOCAL_M0 LOCAL_M1 <<< $(python3 "$STAT_SCRIPT" "$VBENCH_JSON" "$BASE_DIR/$TARGET_DIR")

    # 远程统计
    REMOTE_RESULT=$(ssh -o ConnectTimeout=3 -o BatchMode=yes $REMOTE_HOST "python3 - '$VBENCH_JSON' '$REMOTE_BASE_DIR/$TARGET_DIR'" << 'RPYEOF' 2>/dev/null
import json, os, sys
json_path, output_dir = sys.argv[1], sys.argv[2]
with open(json_path) as f: data = json.load(f)
mid = len(data) // 2
m0_prompts = set(item["prompt_en"] for item in data[:mid])
m0_done, m1_done = 0, 0
if os.path.exists(output_dir):
    for f in os.listdir(output_dir):
        if not f.endswith('.mp4'): continue
        name = f.replace('.mp4', '')
        prompt = name.rsplit('-', 1)[0] if name[-1:].isdigit() and len(name) > 2 and name[-2] == '-' else name
        if prompt in m0_prompts: m0_done += 1
        else: m1_done += 1
print(m0_done, m1_done)
RPYEOF
)

    if [ $? -eq 0 ] && [ -n "$REMOTE_RESULT" ]; then
        read REMOTE_M0 REMOTE_M1 <<< "$REMOTE_RESULT"
        REMOTE_STATUS="在线"
    else
        REMOTE_M0=0; REMOTE_M1=0
        REMOTE_STATUS="离线"
    fi

    # 真实进度: 取两边最大值 (因为文件会同步)
    M0_DONE=$LOCAL_M0; [ $REMOTE_M0 -gt $LOCAL_M0 ] && M0_DONE=$REMOTE_M0
    M1_DONE=$LOCAL_M1; [ $REMOTE_M1 -gt $LOCAL_M1 ] && M1_DONE=$REMOTE_M1

    TOTAL_DONE=$((M0_DONE + M1_DONE))
    TOTAL_REMAINING=$((GOAL_TOTAL - TOTAL_DONE))
    [ $TOTAL_REMAINING -lt 0 ] && TOTAL_REMAINING=0

    M0_PCT=$(awk "BEGIN {printf \"%.1f\", ($M0_DONE/$GOAL_PER_MACHINE)*100}")
    M1_PCT=$(awk "BEGIN {printf \"%.1f\", ($M1_DONE/$GOAL_PER_MACHINE)*100}")
    TOTAL_PCT=$(awk "BEGIN {printf \"%.1f\", ($TOTAL_DONE/$GOAL_TOTAL)*100}")

    if [ "$FIRST_RUN" = true ]; then
        TOTAL_START_COUNT=$TOTAL_DONE
        FIRST_RUN=false
    fi

    ELAPSED=$((CURRENT_TIME - START_TIME))
    MADE=$((TOTAL_DONE - TOTAL_START_COUNT))

    if [ $MADE -gt 0 ] && [ $ELAPSED -gt 0 ]; then
        SPEED=$(awk "BEGIN {printf \"%.1f\", ($MADE/$ELAPSED)*60}")
        ETA_SEC=$(awk "BEGIN {printf \"%.0f\", $TOTAL_REMAINING/($MADE/$ELAPSED)}")
        ETA_STR=$(format_time $ETA_SEC)
    else
        SPEED="--"
        ETA_STR="等待生成..."
    fi

    clear
    echo "========================================================================"
    echo "       VBench I2V 多机监控 - $(date '+%H:%M:%S')  [远程:$REMOTE_STATUS]"
    echo "========================================================================"
    printf "%-16s | %-10s | %-10s | %-10s\n" "任务" "已完成" "剩余" "进度"
    echo "------------------------------------------------------------------------"
    printf "\033[36m%-16s\033[0m | \033[32m%-10s\033[0m | %-10s | %s%%\n" \
        "M0-本地" "$M0_DONE" "$((GOAL_PER_MACHINE-M0_DONE))" "$M0_PCT"
    printf "\033[35m%-16s\033[0m | \033[32m%-10s\033[0m | %-10s | %s%%\n" \
        "M1-远程" "$M1_DONE" "$((GOAL_PER_MACHINE-M1_DONE))" "$M1_PCT"
    echo "------------------------------------------------------------------------"
    printf "\033[1m%-16s\033[0m | \033[32m%-10s\033[0m | %-10s | %s%%\n" \
        "合计" "$TOTAL_DONE" "$TOTAL_REMAINING" "$TOTAL_PCT"
    echo "========================================================================"
    echo "速度: $SPEED/min | 运行: $(format_time $ELAPSED) | ETA: $ETA_STR"
    echo "========================================================================"

    sleep $REFRESH_RATE
done
