#!/bin/bash
# Run 3 video generation methods on 8 GPUs and upload to HuggingFace
# FastVideo (GPU 0-3), Krea (GPU 4-5), LightX2V (GPU 6-7)

set -e

cd /root/data/video-gen-related

source /root/FGA/miniconda3/bin/activate fastvideo

python run_and_upload.py

echo "Done!"

bash /root/data/video-gen-related/run_vbench.sh
