#!/bin/bash
###############################################################################
# Helios VBench All-in-One Script
#
# 一键完成: 环境安装 -> 权重下载 -> T2V生成 -> I2V生成
# 8卡并行生成, 每个worker独立运行
#
# 用法:
#   bash helios_vbench_all_in_one.sh
#
# 需要的文件 (会自动下载):
#   - Helios-Distilled 权重
#   - VBench_aug_full_info.json (T2V prompts)
#   - vbench2_i2v_aug_full_info.json (I2V prompts + images)
###############################################################################
set -e

# ===================== CONFIG =====================
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_PATH="${CONDA_PATH:-$(conda info --base)}"
ENV_NAME="helios"
MODEL_ID="BestWishYsh/Helios-Distilled"
MODEL_DIR="${BASE_DIR}/Helios-Distilled"

# VBench paths
VBENCH_T2V_JSON="${BASE_DIR}/VBench_aug_full_info.json"
VBENCH_I2V_JSON="${BASE_DIR}/vbench2_i2v_aug_full_info.json"
I2V_IMAGE_DIR="${BASE_DIR}/datasets/vbench_i2v/crop"

# Output
OUTPUT_BASE="${BASE_DIR}/outputs/vbench"
T2V_OUTPUT="${OUTPUT_BASE}/helios_t2v"
I2V_OUTPUT="${OUTPUT_BASE}/helios_i2v"

NUM_GPUS=8
NUM_SAMPLES=5
# ==================================================

echo "=============================================="
echo " Helios VBench All-in-One"
echo "=============================================="
echo "BASE_DIR: ${BASE_DIR}"
echo "CONDA:   ${CONDA_PATH}"
echo ""

# ===================== STEP 1: Environment =====================
echo "[Step 1/5] Setting up conda environment..."

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  -> Environment '${ENV_NAME}' exists, checking packages..."
else
    echo "  -> Creating environment '${ENV_NAME}'..."
    conda create -n ${ENV_NAME} python=3.11 -y
fi

# Activate
source "${CONDA_PATH}/bin/activate" ${ENV_NAME}

# Install deps
echo "  -> Installing packages..."
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126 -q 2>/dev/null || true
pip install "git+https://github.com/huggingface/diffusers.git" -q 2>/dev/null || true
pip install transformers accelerate safetensors einops ftfy imageio imageio-ffmpeg -q 2>/dev/null || true
pip install vbench -q 2>/dev/null || true

echo "  -> Verifying..."
python -c "from diffusers import HeliosPyramidPipeline; print('  -> diffusers OK')"
python -c "import vbench; print('  -> vbench OK')"
echo "[Step 1] Done."
echo ""

# ===================== STEP 2: Download Weights =====================
echo "[Step 2/5] Downloading model weights..."

if [ -f "${MODEL_DIR}/model_index.json" ]; then
    echo "  -> Helios-Distilled already downloaded."
else
    echo "  -> Downloading Helios-Distilled (~60GB)..."
    huggingface-cli download ${MODEL_ID} --local-dir "${MODEL_DIR}"
fi

# Download VBench metadata if not present
if [ ! -f "${VBENCH_T2V_JSON}" ]; then
    echo "  -> VBench_aug_full_info.json not found, please provide it."
    exit 1
fi

echo "[Step 2] Done."
echo ""

# ===================== STEP 3: Generate T2V Worker Script =====================
echo "[Step 3/5] Generating T2V videos (8 GPU workers)..."

mkdir -p "${T2V_OUTPUT}"

cat > "${BASE_DIR}/_helios_t2v_worker.py" << 'WORKER_EOF'
"""Helios T2V VBench worker - generates videos for a subset of prompts."""
import json
import os
import sys
import argparse
from pathlib import Path

import torch
from diffusers import AutoModel, HeliosPyramidPipeline
from diffusers.utils import export_to_video

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
    "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
    "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--vbench_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    # Load prompts
    with open(args.vbench_json) as f:
        vbench_data = json.load(f)

    num_prompts = len(vbench_data)
    total_samples = num_prompts * args.num_samples

    # Build sample list: (prompt_idx, sample_idx)
    all_indices = []
    for p_idx in range(num_prompts):
        for s_idx in range(args.num_samples):
            all_indices.append((p_idx, s_idx))

    # Rotate for load balancing
    offset = args.worker_id * (total_samples // args.num_workers)
    all_indices = all_indices[offset:] + all_indices[:offset]

    # Load pipeline
    print(f"[Worker {args.worker_id}] Loading Helios-Distilled...")
    vae = AutoModel.from_pretrained(args.model_dir, subfolder="vae", torch_dtype=torch.float32)
    pipeline = HeliosPyramidPipeline.from_pretrained(
        args.model_dir, vae=vae, torch_dtype=torch.bfloat16
    )
    pipeline.to("cuda")
    print(f"[Worker {args.worker_id}] Pipeline loaded. Processing {total_samples} samples...")

    os.makedirs(args.output_dir, exist_ok=True)
    generated = 0

    for p_idx, s_idx in all_indices:
        item = vbench_data[p_idx]
        prompt_en = item["prompt_en"]
        aug_prompt = item.get("aug_prompt_en", prompt_en)
        target_name = f"{prompt_en}-{s_idx}.mp4"
        target_path = Path(args.output_dir) / target_name
        lock_path = Path(args.output_dir) / f".{prompt_en}-{s_idx}.lock"

        # Skip if exists
        if target_path.exists():
            continue

        # Lock file to prevent duplicate work
        try:
            lock_path.touch(exist_ok=False)
        except FileExistsError:
            continue

        try:
            seed = s_idx * 1000 + p_idx
            generator = torch.Generator(device="cuda").manual_seed(seed)

            output = pipeline(
                prompt=aug_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_frames=132,
                height=384,
                width=640,
                pyramid_num_inference_steps_list=[2, 2, 2],
                guidance_scale=1.0,
                is_amplify_first_chunk=True,
                generator=generator,
            )

            export_to_video(output.frames[0], str(target_path), fps=24)
            generated += 1

            if generated % 10 == 0:
                done = len([f for f in os.listdir(args.output_dir) if f.endswith('.mp4')])
                print(f"[Worker {args.worker_id}] Generated {generated} (total {done}/{total_samples})")
        except Exception as e:
            print(f"[Worker {args.worker_id}] Error: {e}")
        finally:
            lock_path.unlink(missing_ok=True)

    print(f"[Worker {args.worker_id}] Done. Generated {generated} videos.")


if __name__ == "__main__":
    main()
WORKER_EOF

# Launch T2V workers
echo "  -> Launching ${NUM_GPUS} T2V workers..."
PIDS=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=${gpu_id} python "${BASE_DIR}/_helios_t2v_worker.py" \
        --worker_id ${gpu_id} \
        --num_workers ${NUM_GPUS} \
        --model_dir "${MODEL_DIR}" \
        --vbench_json "${VBENCH_T2V_JSON}" \
        --output_dir "${T2V_OUTPUT}" \
        --num_samples ${NUM_SAMPLES} \
        > "${OUTPUT_BASE}/helios_t2v_worker${gpu_id}.log" 2>&1 &
    PIDS+=($!)
    echo "    GPU ${gpu_id}: PID $!"
done

echo "  -> Waiting for T2V generation to complete..."
FAIL=0
for pid in "${PIDS[@]}"; do
    wait $pid || FAIL=$((FAIL + 1))
done

T2V_COUNT=$(find "${T2V_OUTPUT}" -name "*.mp4" | wc -l)
T2V_EXPECTED=$(python -c "import json; d=json.load(open('${VBENCH_T2V_JSON}')); print(len(d) * ${NUM_SAMPLES})")
echo "[Step 3] T2V Done. ${T2V_COUNT}/${T2V_EXPECTED} videos. Failed workers: ${FAIL}"
echo ""

# ===================== STEP 4: Generate I2V Worker Script =====================
echo "[Step 4/5] Generating I2V videos (8 GPU workers)..."

if [ ! -f "${VBENCH_I2V_JSON}" ]; then
    echo "  -> WARNING: ${VBENCH_I2V_JSON} not found, skipping I2V."
else

mkdir -p "${I2V_OUTPUT}"

cat > "${BASE_DIR}/_helios_i2v_worker.py" << 'WORKER_EOF'
"""Helios I2V VBench worker - generates videos from images."""
import json
import os
import sys
import argparse
from pathlib import Path

import torch
from diffusers import AutoModel, HeliosPyramidPipeline
from diffusers.utils import export_to_video, load_image

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
    "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
    "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--vbench_json", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    with open(args.vbench_json) as f:
        vbench_data = json.load(f)

    num_prompts = len(vbench_data)
    total_samples = num_prompts * args.num_samples

    all_indices = []
    for p_idx in range(num_prompts):
        for s_idx in range(args.num_samples):
            all_indices.append((p_idx, s_idx))

    offset = args.worker_id * (total_samples // args.num_workers)
    all_indices = all_indices[offset:] + all_indices[:offset]

    print(f"[Worker {args.worker_id}] Loading Helios-Distilled for I2V...")
    vae = AutoModel.from_pretrained(args.model_dir, subfolder="vae", torch_dtype=torch.float32)
    pipeline = HeliosPyramidPipeline.from_pretrained(
        args.model_dir, vae=vae, torch_dtype=torch.bfloat16
    )
    pipeline.to("cuda")
    print(f"[Worker {args.worker_id}] Pipeline loaded.")

    os.makedirs(args.output_dir, exist_ok=True)
    generated = 0

    for p_idx, s_idx in all_indices:
        item = vbench_data[p_idx]
        prompt_en = item["prompt_en"]
        aug_prompt = item.get("aug_prompt_en", prompt_en)
        image_name = item.get("image_name", "")
        target_name = f"{prompt_en}-{s_idx}.mp4"
        target_path = Path(args.output_dir) / target_name
        lock_path = Path(args.output_dir) / f".{prompt_en}-{s_idx}.lock"

        if target_path.exists():
            continue

        try:
            lock_path.touch(exist_ok=False)
        except FileExistsError:
            continue

        try:
            # Find source image
            image_path = os.path.join(args.image_dir, image_name)
            if not os.path.exists(image_path):
                # Try without subdirectory
                for root, _, files in os.walk(args.image_dir):
                    if image_name in files:
                        image_path = os.path.join(root, image_name)
                        break

            if not os.path.exists(image_path):
                print(f"[Worker {args.worker_id}] Image not found: {image_name}, skipping")
                continue

            image = load_image(image_path).resize((640, 384))
            seed = s_idx * 1000 + p_idx
            generator = torch.Generator(device="cuda").manual_seed(seed)

            output = pipeline(
                prompt=aug_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                image=image,
                num_frames=132,
                height=384,
                width=640,
                pyramid_num_inference_steps_list=[2, 2, 2],
                guidance_scale=1.0,
                is_amplify_first_chunk=True,
                generator=generator,
            )

            export_to_video(output.frames[0], str(target_path), fps=24)
            generated += 1

            if generated % 10 == 0:
                done = len([f for f in os.listdir(args.output_dir) if f.endswith('.mp4')])
                print(f"[Worker {args.worker_id}] I2V Generated {generated} (total {done}/{total_samples})")
        except Exception as e:
            print(f"[Worker {args.worker_id}] Error on {prompt_en}: {e}")
        finally:
            lock_path.unlink(missing_ok=True)

    print(f"[Worker {args.worker_id}] I2V Done. Generated {generated} videos.")


if __name__ == "__main__":
    main()
WORKER_EOF

# Launch I2V workers
echo "  -> Launching ${NUM_GPUS} I2V workers..."
PIDS=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=${gpu_id} python "${BASE_DIR}/_helios_i2v_worker.py" \
        --worker_id ${gpu_id} \
        --num_workers ${NUM_GPUS} \
        --model_dir "${MODEL_DIR}" \
        --vbench_json "${VBENCH_I2V_JSON}" \
        --image_dir "${I2V_IMAGE_DIR}" \
        --output_dir "${I2V_OUTPUT}" \
        --num_samples ${NUM_SAMPLES} \
        > "${OUTPUT_BASE}/helios_i2v_worker${gpu_id}.log" 2>&1 &
    PIDS+=($!)
    echo "    GPU ${gpu_id}: PID $!"
done

echo "  -> Waiting for I2V generation to complete..."
FAIL=0
for pid in "${PIDS[@]}"; do
    wait $pid || FAIL=$((FAIL + 1))
done

I2V_COUNT=$(find "${I2V_OUTPUT}" -name "*.mp4" | wc -l)
echo "[Step 4] I2V Done. ${I2V_COUNT} videos. Failed workers: ${FAIL}"

fi  # end I2V check
echo ""

# ===================== STEP 5: Summary =====================
echo "=============================================="
echo " All Done!"
echo "=============================================="
echo ""
echo "T2V videos: ${T2V_OUTPUT}  ($(find "${T2V_OUTPUT}" -name '*.mp4' 2>/dev/null | wc -l) files)"
echo "I2V videos: ${I2V_OUTPUT}  ($(find "${I2V_OUTPUT}" -name '*.mp4' 2>/dev/null | wc -l) files)"
echo ""
echo "To run VBench evaluation:"
echo "  source ${CONDA_PATH}/bin/activate ${ENV_NAME}"
echo "  torchrun --nproc_per_node=8 ${BASE_DIR}/run_vbench.py --video_dir ${T2V_OUTPUT}"
echo ""
echo "Cleanup temp files:"
echo "  rm -f ${BASE_DIR}/_helios_t2v_worker.py ${BASE_DIR}/_helios_i2v_worker.py"
echo "  find ${T2V_OUTPUT} ${I2V_OUTPUT} -name '*.lock' -delete"
