#!/bin/bash
# VBench 8卡并行评估脚本
#
# 用法:
#   bash run_vbench.sh /path/to/video_dir
#
# 示例:
#   bash run_vbench.sh /root/data/rcm/output/vbench_results/step2
#   bash run_vbench.sh /root/data/rcm/output/vbench_results/step4

# if [ -z "$1" ]; then
#     echo "用法: bash run_vbench.sh <video_dir>"
#     echo "示例: bash run_vbench.sh /root/data/rcm/output/vbench_results/step4"
#     exit 1
# fi

VIDEO_DIR=/root/data/rcm/output/vbench_results/step2
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

echo "=========================================="
echo "VBench 评估"
echo "视频目录: $VIDEO_DIR"
echo "=========================================="

cd "$SCRIPT_DIR"
torchrun --nproc_per_node=8 run_vbench.py --video_dir "$VIDEO_DIR"
