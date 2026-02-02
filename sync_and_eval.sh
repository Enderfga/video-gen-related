# #!/bin/bash

# # 同步远程数据并运行 VBench I2V 评估

# REMOTE_HOST="root@31.22.104.21"
LOCAL_DIR="/root/data/video-gen-related/outputs/vbench/fastvideo_i2v"
# REMOTE_DIR="/root/data/video-gen-related/outputs/vbench/fastvideo_i2v"
CONDA_PATH="/root/FGA/miniconda3"

# echo "========================================================================"
# echo "       同步 & VBench I2V 评估"
# echo "========================================================================"

# # Step 1: 同步远程数据
# echo ""
# echo "[Step 1] 从远程同步视频文件..."
# echo "  远程: $REMOTE_HOST:$REMOTE_DIR"
# echo "  本地: $LOCAL_DIR"
# echo ""

# rsync -avz --progress "$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"

# if [ $? -ne 0 ]; then
#     echo "同步失败!"
#     exit 1
# fi

# 统计文件数
LOCAL_COUNT=$(find "$LOCAL_DIR" -name "*.mp4" | wc -l)
echo ""
echo "同步完成! 本地视频总数: $LOCAL_COUNT"

# Step 2: 检查是否达到目标数量
TARGET=5590
if [ $LOCAL_COUNT -lt $TARGET ]; then
    echo ""
    echo "警告: 视频数量 ($LOCAL_COUNT) 未达到目标 ($TARGET)"
    read -p "是否继续评估? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 0
    fi
fi

# Step 3: 运行 VBench I2V 评估
echo ""
echo "[Step 2] 运行 VBench I2V 评估..."
echo ""

source "$CONDA_PATH/bin/activate" fastvideo

python3 << 'PYEOF'
import os
import json
import sys

os.environ['VBENCH_CACHE_DIR'] = '/root/data/FAR-World/experiments/pretrained_models/vbench'
sys.path.insert(0, '/root/data/FAR-World')

from vbench2_beta_i2v import VBenchI2V

VIDEO_DIR = "/root/data/video-gen-related/outputs/vbench/fastvideo_i2v"
OUTPUT_DIR = "/root/data/video-gen-related/outputs/vbench/eval_results"

DIMENSIONS = [
    'camera_motion', 'i2v_subject', 'i2v_background',
    'subject_consistency', 'motion_smoothness', 'background_consistency',
    'dynamic_degree', 'aesthetic_quality', 'imaging_quality'
]

METRICS_NORMALIZATION_RANGES = {
    'subject_consistency': [0.1462, 1.0],
    'motion_smoothness': [0.706, 0.9975],
    'temporal_flickering': [0.6293, 1.0],
    'background_consistency': [0.2615, 1.0],
    'scene': [0.0, 0.8222],
    'appearance_style': [0.0009, 0.2855],
    'temporal_style': [0.0, 0.364],
    'overall_consistency': [0.0, 0.364],
    'i2v_subject': [0.1462, 1.0],
    'i2v_background': [0.2615, 1.0],
    'dynamic_degree': [0.0, 1.0],
    'aesthetic_quality': [0.0, 1.0],
    'imaging_quality': [0.0, 1.0],
    'camera_motion': [0.0, 1.0],
}

def norm(metric, key):
    range_ = METRICS_NORMALIZATION_RANGES.get(key, [0.0, 1.0])
    metric = max(metric, range_[0])
    metric = min(metric, range_[1])
    metric = (metric - range_[0]) / (range_[1] - range_[0])
    return metric

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Video directory: {VIDEO_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Dimensions: {DIMENSIONS}")
print()

vbench = VBenchI2V(
    device='cuda:0',
    full_json_dir='/root/data/FAR-World/assets/data/meta/vbench/vbench2_i2v_full_info.json',
    output_path=OUTPUT_DIR
)

print("Running evaluation...")
for dim in DIMENSIONS:
    print(f"  Evaluating: {dim}")
    vbench.evaluate(
        videos_path=VIDEO_DIR,
        name="fastvideo_i2v",
        dimension=[dim],
    )

# Load and normalize results
print("\n" + "=" * 60)
print("VBench I2V 评估结果")
print("=" * 60)

normalized_results = {}
for dim in DIMENSIONS:
    results_file = os.path.join(OUTPUT_DIR, f"{dim}_eval_results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
        # 取平均分
        if isinstance(data, list) and len(data) > 0:
            scores = [item[1] for item in data if isinstance(item, list) and len(item) > 1]
            if scores:
                raw_score = sum(scores) / len(scores)
                norm_score = norm(raw_score, dim)
                normalized_results[dim] = norm_score
                print(f"{dim:25s}: {norm_score:.4f} (raw: {raw_score:.4f})")

# 计算总分
if normalized_results:
    total_score = sum(normalized_results.values()) / len(normalized_results)
    print("-" * 60)
    print(f"{'Total Score':25s}: {total_score:.4f}")

    # 保存归一化结果
    norm_results_file = os.path.join(OUTPUT_DIR, "fastvideo_i2v_normalized_results.json")
    with open(norm_results_file, 'w') as f:
        json.dump(normalized_results, f, indent=2)
    print(f"\n归一化结果已保存到: {norm_results_file}")
else:
    print("没有找到评估结果")

print("\n评估完成!")
PYEOF

echo ""
echo "========================================================================"
echo "       完成!"
echo "========================================================================"
