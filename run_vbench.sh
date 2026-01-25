#!/bin/bash
# VBench evaluation for FastVideo, Krea, LightX2V
#
# GPU allocation:
# - GPU 0-3: FastVideo (4 workers)
# - GPU 4-5: Krea (2 workers)
# - GPU 6-7: LightX2V (2 workers)
#
# After generation: VBench scoring with rcm env, then upload to Enderfga/vbench

set -e

echo "=========================================="
echo "VBench Evaluation"
echo "  FastVideo -> GPU 0-3 (4 workers)"
echo "  Krea      -> GPU 4-5 (2 workers)"
echo "  LightX2V  -> GPU 6-7 (2 workers)"
echo "  Prompts: 946"
echo "  Samples per prompt: 5"
echo "  Total: 4730 videos per method"
echo "=========================================="

source ~/FGA/miniconda3/etc/profile.d/conda.sh
conda activate fastvideo

cd /root/data/video-gen-related

python run_vbench.py

echo "=========================================="
echo "VBench complete! Results uploaded to:"
echo "https://huggingface.co/datasets/Enderfga/vbench"
echo "=========================================="
