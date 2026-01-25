#!/bin/bash

# ================= 配置区域 =================
BASE_DIR="/root/data/video-gen-related/outputs/vbench"
TARGET_DIRS=("fastvideo" "lightx2v" "krea")
GOAL_PER_FOLDER=$((946 * 5))  # 单个文件夹的目标 (4730)

# 如果这三个文件夹是合作完成同一个 4730 的任务，把下面设为 true
# 如果是三个模型各自都要跑 4730 (通常是这种情况)，设为 false
IS_SHARED_GOAL=false 

REFRESH_RATE=5
# ===========================================

# 检查目录
for dir in "${TARGET_DIRS[@]}"; do
    mkdir -p "$BASE_DIR/$dir" # 防止目录不存在报错
done

# 初始化
START_TIME=$(date +%s)
FIRST_RUN=true
TOTAL_START_COUNT=0

# 时间格式化函数
format_time() {
    local T=$1
    local H=$((T/3600))
    local M=$(( (T%3600)/60 ))
    local S=$((T%60))
    printf "%02d:%02d:%02d" $H $M $S
}

while true; do
    CURRENT_TIME=$(date +%s)
    
    # 准备数据收集
    GRAND_TOTAL_DONE=0
    GRAND_TOTAL_LOCK=0
    
    # 清屏并打印表头
    clear
    echo "========================================================================"
    echo "       VBench 多模型生成监控 - $(date "+%H:%M:%S")"
    echo "========================================================================"
    printf "%-12s | %-10s | %-10s | %-10s | %-15s\n" "模型/文件夹" "已完成" "进行中(.lock)" "剩余" "进度"
    echo "------------------------------------------------------------------------"

    # 循环检查每个文件夹
    for dir in "${TARGET_DIRS[@]}"; do
        # 统计当前文件夹
        DONE=$(find "$BASE_DIR/$dir" -type f -not -name "*.lock" 2>/dev/null | wc -l)
        LOCK=$(find "$BASE_DIR/$dir" -type f -name "*.lock" 2>/dev/null | wc -l)
        
        # 计算该文件夹剩余
        if [ "$IS_SHARED_GOAL" = true ]; then
             # 共享目标模式下，剩余数是全局的，这里只显示它贡献了多少
             REMAINING_THIS="-"
             PCT="-"
        else
             REMAINING_THIS=$((GOAL_PER_FOLDER - DONE))
             if [ $REMAINING_THIS -lt 0 ]; then REMAINING_THIS=0; fi
             PCT=$(awk "BEGIN {printf \"%.2f\", ($DONE/$GOAL_PER_FOLDER)*100}")
        fi

        # 打印一行数据
        if [ "$IS_SHARED_GOAL" = true ]; then
            printf "%-12s | \033[32m%-10s\033[0m | \033[33m%-10s\033[0m | %-10s | %-15s\n" "$dir" "$DONE" "$LOCK" "$REMAINING_THIS" "$PCT"
        else
            # 进度条可视化
            BAR_LEN=$(awk "BEGIN {printf \"%.0f\", ($PCT/100)*10}")
            BAR_STR=""
            for ((i=0; i<BAR_LEN; i++)); do BAR_STR="${BAR_STR}#"; done
            printf "%-12s | \033[32m%-10s\033[0m | \033[33m%-10s\033[0m | %-10s | %-6s (%s)\n" "$dir" "$DONE" "$LOCK" "$REMAINING_THIS" "${PCT}%" "$BAR_STR"
        fi

        # 累加总数
        GRAND_TOTAL_DONE=$((GRAND_TOTAL_DONE + DONE))
        GRAND_TOTAL_LOCK=$((GRAND_TOTAL_LOCK + LOCK))
    done

    # --- 总体统计 ---
    if [ "$IS_SHARED_GOAL" = true ]; then
        TOTAL_TARGET=$GOAL_PER_FOLDER
    else
        TOTAL_TARGET=$((GOAL_PER_FOLDER * ${#TARGET_DIRS[@]}))
    fi

    TOTAL_REMAINING=$((TOTAL_TARGET - GRAND_TOTAL_DONE))
    if [ $TOTAL_REMAINING -lt 0 ]; then TOTAL_REMAINING=0; fi
    TOTAL_PCT=$(awk "BEGIN {printf \"%.2f\", ($GRAND_TOTAL_DONE/$TOTAL_TARGET)*100}")

    # --- 速度计算 ---
    if [ "$FIRST_RUN" = true ]; then
        TOTAL_START_COUNT=$GRAND_TOTAL_DONE
        FIRST_RUN=false
        sleep 1
        continue
    fi

    ELAPSED=$((CURRENT_TIME - START_TIME))
    MADE_SINCE_START=$((GRAND_TOTAL_DONE - TOTAL_START_COUNT))

    if [ $MADE_SINCE_START -gt 0 ] && [ $ELAPSED -gt 0 ]; then
        # 整体吞吐量
        SPEED_MIN=$(awk "BEGIN {printf \"%.2f\", ($MADE_SINCE_START/$ELAPSED)*60}")
        SEC_PER_ITEM=$(awk "BEGIN {printf \"%.2f\", $ELAPSED/$MADE_SINCE_START}")
        
        if [ $TOTAL_REMAINING -gt 0 ]; then
            ETA_SEC=$(awk "BEGIN {printf \"%.0f\", $TOTAL_REMAINING * $SEC_PER_ITEM}")
            ETA_STR=$(format_time $ETA_SEC)
        else
            ETA_STR="全部完成"
        fi
    else
        SPEED_MIN="计算中..."
        ETA_STR="等待数据..."
    fi

    echo "------------------------------------------------------------------------"
    echo -e "📊 总进度:     \033[32m$GRAND_TOTAL_DONE\033[0m / $TOTAL_TARGET ($TOTAL_PCT%)"
    echo -e "⚙️  系统总负载: \033[33m$GRAND_TOTAL_LOCK\033[0m 个任务正在并发处理"
    echo "------------------------------------------------------------------------"
    echo "🚀 综合速度:   $SPEED_MIN 个/分钟"
    echo "⏱️  预计全部完成: $ETA_STR"
    echo "========================================================================"
    echo "按 Ctrl+C 退出"
    
    sleep $REFRESH_RATE
done