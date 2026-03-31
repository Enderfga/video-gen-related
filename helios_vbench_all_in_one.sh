#!/bin/bash
###############################################################################
# Helios VBench All-in-One Script
#
# 一键完成: 环境安装 -> 权重下载 -> I2V图片下载 -> T2V生成 -> I2V生成
# 8卡并行生成, 每个worker独立运行
#
# 用法:
#   bash helios_vbench_all_in_one.sh
###############################################################################
set -e

# ===================== CONFIG =====================
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${BASE_DIR}/.venv_helios"
MODEL_ID="BestWishYsh/Helios-Distilled"
MODEL_DIR="${BASE_DIR}/Helios-Distilled"

# VBench data
VBENCH_T2V_JSON="${BASE_DIR}/VBench_aug_full_info.json"
VBENCH_I2V_JSON="${BASE_DIR}/vbench2_i2v_full_info.json"
I2V_DATA_DIR="${BASE_DIR}/datasets/vbench_i2v/data"
I2V_IMAGE_DIR="${I2V_DATA_DIR}/crop"

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
echo ""

# ===================== STEP 1: Python venv =====================
echo "[Step 1/6] Setting up Python virtual environment..."

# Setup Python environment with conda (need Python 3.11 for tokenizers wheels)
CONDA_BASE="$(conda info --base 2>/dev/null || echo "")"

if [ -n "${CONDA_BASE}" ]; then
    echo "  -> Using conda at ${CONDA_BASE}"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"

    if ! conda env list | grep -q "^helios "; then
        echo "  -> Creating conda env 'helios' with Python 3.11..."
        conda create -n helios python=3.11 -y
    fi
    conda activate helios
    PY="$(conda run -n helios which python)"
    PIP="$(conda run -n helios which pip)"
else
    echo "  -> No conda found, using venv..."
    if [ ! -f "${VENV_DIR}/bin/python" ]; then
        python3 -m venv "${VENV_DIR}"
    fi
    source "${VENV_DIR}/bin/activate"
    PY="python"
    PIP="pip"
fi

echo "  -> Python: $($PY --version)"

# Unset PIP_CONSTRAINT which can break installs
unset PIP_CONSTRAINT

echo "  -> Installing packages..."
$PIP install -q --upgrade pip setuptools wheel

# Ensure Rust is available (needed by tokenizers if no prebuilt wheel)
if [ -f "$HOME/.cargo/env" ]; then source "$HOME/.cargo/env"; fi
export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v rustc &> /dev/null; then
    echo "  -> Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# NOTE: If env already created, skip to Step 2 by commenting out the pip lines below.
# $PIP install ...  lines can be skipped on re-run if packages are already installed.

$PIP install -q torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -3
$PIP install -q "git+https://github.com/huggingface/diffusers.git" 2>&1 | tail -3
$PIP install -q transformers accelerate safetensors einops ftfy imageio imageio-ffmpeg tokenizers 2>&1 | tail -3
# vbench pins old transformers, install it without deps then fix
$PIP install -q --no-deps vbench 2>&1 | tail -3
$PIP install -q gdown huggingface-hub pyiqa decord opencv-python scipy scikit-learn scikit-image 2>&1 | tail -3
$PIP install -q openai-clip timm easydict omegaconf 2>&1 | tail -3

echo "  -> Verifying..."
$PY -c "from diffusers import HeliosPyramidPipeline; print('  -> diffusers OK')"
$PY -c "import vbench; print('  -> vbench OK')"
echo "[Step 1] Done."
echo ""

# ===================== STEP 2: Download Model Weights =====================
echo "[Step 2/6] Downloading Helios-Distilled weights..."

if [ -f "${MODEL_DIR}/model_index.json" ]; then
    echo "  -> Already downloaded."
else
    echo "  -> Downloading (~60GB)..."
    huggingface-cli download ${MODEL_ID} --local-dir "${MODEL_DIR}"
fi

echo "[Step 2] Done."
echo ""

# ===================== STEP 3: Download VBench I2V Data =====================
echo "[Step 3/6] Preparing VBench I2V data..."

# T2V json should be in repo
if [ ! -f "${VBENCH_T2V_JSON}" ]; then
    echo "  -> ERROR: ${VBENCH_T2V_JSON} not found!"
    exit 1
fi
echo "  -> T2V json: OK ($($PY -c "import json; print(len(json.load(open('${VBENCH_T2V_JSON}'))))" ) prompts)"

# I2V json should be in repo
if [ ! -f "${VBENCH_I2V_JSON}" ]; then
    echo "  -> ERROR: ${VBENCH_I2V_JSON} not found!"
    exit 1
fi
echo "  -> I2V json: OK ($($PY -c "import json; print(len(json.load(open('${VBENCH_I2V_JSON}'))))" ) prompts)"

# Download I2V images if not present
if [ -d "${I2V_IMAGE_DIR}" ] && [ "$(ls -A ${I2V_IMAGE_DIR} 2>/dev/null)" ]; then
    echo "  -> I2V images: already downloaded."
else
    echo "  -> Downloading I2V images from Google Drive..."
    mkdir -p "${I2V_DATA_DIR}"

    # Download crop.zip (pre-cropped images in multiple aspect ratios)
    gdown --id 1Y_JnYnyJ3a6QhiranoX0MQVZFcTDPekZ --output "${I2V_DATA_DIR}/crop.zip"
    unzip -q "${I2V_DATA_DIR}/crop.zip" -d "${I2V_DATA_DIR}"
    rm -f "${I2V_DATA_DIR}/crop.zip"

    echo "  -> I2V images downloaded."
fi

# Determine best aspect ratio directory for Helios (640x384 ≈ 5:3)
# Available: 1-1, 7-4, 8-5, 16-9. Closest to 5:3 is 8-5 (1.6 vs 1.67)
if [ -d "${I2V_IMAGE_DIR}/8-5" ]; then
    I2V_CROP_DIR="${I2V_IMAGE_DIR}/8-5"
elif [ -d "${I2V_IMAGE_DIR}/7-4" ]; then
    I2V_CROP_DIR="${I2V_IMAGE_DIR}/7-4"
else
    I2V_CROP_DIR="${I2V_IMAGE_DIR}"
fi
echo "  -> Using crop dir: ${I2V_CROP_DIR}"

echo "[Step 3] Done."
echo ""

# ===================== STEP 4: T2V Generation =====================
echo "[Step 4/6] Generating T2V videos (8 GPU workers)..."

mkdir -p "${T2V_OUTPUT}"

cat > "${BASE_DIR}/_helios_t2v_worker.py" << 'WORKER_EOF'
"""Helios T2V VBench worker."""
import json, os, argparse
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
    with open(args.vbench_json) as f:
        vbench_data = json.load(f)

    total_samples = len(vbench_data) * args.num_samples
    all_indices = [(p, s) for p in range(len(vbench_data)) for s in range(args.num_samples)]
    offset = args.worker_id * (total_samples // args.num_workers)
    all_indices = all_indices[offset:] + all_indices[:offset]

    print(f"[Worker {args.worker_id}] Loading Helios-Distilled...")
    vae = AutoModel.from_pretrained(args.model_dir, subfolder="vae", torch_dtype=torch.float32)
    pipeline = HeliosPyramidPipeline.from_pretrained(args.model_dir, vae=vae, torch_dtype=torch.bfloat16)
    pipeline.to("cuda")
    print(f"[Worker {args.worker_id}] Loaded. {total_samples} total samples.")

    os.makedirs(args.output_dir, exist_ok=True)
    generated = 0

    for p_idx, s_idx in all_indices:
        item = vbench_data[p_idx]
        prompt_en = item["prompt_en"]
        aug_prompt = item.get("aug_prompt_en", prompt_en)
        target_path = Path(args.output_dir) / f"{prompt_en}-{s_idx}.mp4"
        lock_path = Path(args.output_dir) / f".{prompt_en}-{s_idx}.lock"

        if target_path.exists():
            continue
        try:
            lock_path.touch(exist_ok=False)
        except FileExistsError:
            continue

        try:
            generator = torch.Generator(device="cuda").manual_seed(s_idx * 1000 + p_idx)
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
                print(f"[Worker {args.worker_id}] T2V {generated} done (total {done}/{total_samples})")
        except Exception as e:
            print(f"[Worker {args.worker_id}] Error: {e}")
        finally:
            lock_path.unlink(missing_ok=True)

    print(f"[Worker {args.worker_id}] T2V finished. Generated {generated}.")

if __name__ == "__main__":
    main()
WORKER_EOF

echo "  -> Launching ${NUM_GPUS} T2V workers..."
PIDS=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=${gpu_id} $PY "${BASE_DIR}/_helios_t2v_worker.py" \
        --worker_id ${gpu_id} --num_workers ${NUM_GPUS} \
        --model_dir "${MODEL_DIR}" --vbench_json "${VBENCH_T2V_JSON}" \
        --output_dir "${T2V_OUTPUT}" --num_samples ${NUM_SAMPLES} \
        > "${OUTPUT_BASE}/helios_t2v_w${gpu_id}.log" 2>&1 &
    PIDS+=($!)
    echo "    GPU ${gpu_id}: PID $!"
done

echo "  -> Waiting for T2V to complete..."
FAIL=0
for pid in "${PIDS[@]}"; do wait $pid || FAIL=$((FAIL + 1)); done

T2V_COUNT=$(find "${T2V_OUTPUT}" -name "*.mp4" | wc -l)
T2V_EXPECTED=$($PY -c "import json; print(len(json.load(open('${VBENCH_T2V_JSON}'))) * ${NUM_SAMPLES})")
echo "[Step 4] T2V Done. ${T2V_COUNT}/${T2V_EXPECTED} videos. Failed workers: ${FAIL}"
echo ""

# ===================== STEP 5: I2V Generation =====================
echo "[Step 5/6] Generating I2V videos (8 GPU workers)..."

mkdir -p "${I2V_OUTPUT}"

cat > "${BASE_DIR}/_helios_i2v_worker.py" << WORKER_EOF
"""Helios I2V VBench worker."""
import json, os, argparse
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
I2V_CROP_DIR = "${I2V_CROP_DIR}"

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
    with open(args.vbench_json) as f:
        vbench_data = json.load(f)

    total_samples = len(vbench_data) * args.num_samples
    all_indices = [(p, s) for p in range(len(vbench_data)) for s in range(args.num_samples)]
    offset = args.worker_id * (total_samples // args.num_workers)
    all_indices = all_indices[offset:] + all_indices[:offset]

    print(f"[Worker {args.worker_id}] Loading Helios-Distilled for I2V...")
    vae = AutoModel.from_pretrained(args.model_dir, subfolder="vae", torch_dtype=torch.float32)
    pipeline = HeliosPyramidPipeline.from_pretrained(args.model_dir, vae=vae, torch_dtype=torch.bfloat16)
    pipeline.to("cuda")
    print(f"[Worker {args.worker_id}] Loaded.")

    os.makedirs(args.output_dir, exist_ok=True)
    generated = 0

    for p_idx, s_idx in all_indices:
        item = vbench_data[p_idx]
        prompt_en = item["prompt_en"]
        image_name = item.get("image_name", "")
        target_path = Path(args.output_dir) / f"{prompt_en}-{s_idx}.mp4"
        lock_path = Path(args.output_dir) / f".{prompt_en}-{s_idx}.lock"

        if target_path.exists():
            continue
        try:
            lock_path.touch(exist_ok=False)
        except FileExistsError:
            continue

        try:
            # Find image in crop directories
            image_path = None
            for root, _, files in os.walk(I2V_CROP_DIR):
                if image_name in files:
                    image_path = os.path.join(root, image_name)
                    break
            if image_path is None:
                print(f"[Worker {args.worker_id}] Image not found: {image_name}, skip")
                continue

            image = load_image(image_path).resize((640, 384))
            generator = torch.Generator(device="cuda").manual_seed(s_idx * 1000 + p_idx)
            output = pipeline(
                prompt=prompt_en,
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
                print(f"[Worker {args.worker_id}] I2V {generated} done (total {done}/{total_samples})")
        except Exception as e:
            print(f"[Worker {args.worker_id}] Error on {prompt_en}: {e}")
        finally:
            lock_path.unlink(missing_ok=True)

    print(f"[Worker {args.worker_id}] I2V finished. Generated {generated}.")

if __name__ == "__main__":
    main()
WORKER_EOF

echo "  -> Launching ${NUM_GPUS} I2V workers..."
PIDS=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=${gpu_id} $PY "${BASE_DIR}/_helios_i2v_worker.py" \
        --worker_id ${gpu_id} --num_workers ${NUM_GPUS} \
        --model_dir "${MODEL_DIR}" --vbench_json "${VBENCH_I2V_JSON}" \
        --output_dir "${I2V_OUTPUT}" --num_samples ${NUM_SAMPLES} \
        > "${OUTPUT_BASE}/helios_i2v_w${gpu_id}.log" 2>&1 &
    PIDS+=($!)
    echo "    GPU ${gpu_id}: PID $!"
done

echo "  -> Waiting for I2V to complete..."
FAIL=0
for pid in "${PIDS[@]}"; do wait $pid || FAIL=$((FAIL + 1)); done

I2V_COUNT=$(find "${I2V_OUTPUT}" -name "*.mp4" | wc -l)
I2V_EXPECTED=$($PY -c "import json; print(len(json.load(open('${VBENCH_I2V_JSON}'))) * ${NUM_SAMPLES})")
echo "[Step 5] I2V Done. ${I2V_COUNT}/${I2V_EXPECTED} videos. Failed workers: ${FAIL}"
echo ""

# ===================== STEP 6: Summary & Cleanup =====================
echo "[Step 6/6] Cleanup & Summary"

# Remove lock files
find "${T2V_OUTPUT}" "${I2V_OUTPUT}" -name "*.lock" -delete 2>/dev/null
# Remove temp worker scripts
rm -f "${BASE_DIR}/_helios_t2v_worker.py" "${BASE_DIR}/_helios_i2v_worker.py"

echo ""
echo "=============================================="
echo " All Done!"
echo "=============================================="
echo ""
echo "T2V: ${T2V_OUTPUT}  ($(find "${T2V_OUTPUT}" -name '*.mp4' 2>/dev/null | wc -l) videos)"
echo "I2V: ${I2V_OUTPUT}  ($(find "${I2V_OUTPUT}" -name '*.mp4' 2>/dev/null | wc -l) videos)"
echo ""
echo "Run VBench T2V evaluation:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  torchrun --nproc_per_node=8 ${BASE_DIR}/run_vbench.py --video_dir ${T2V_OUTPUT}"
echo ""
echo "Run VBench I2V evaluation:"
echo "  torchrun --nproc_per_node=8 ${BASE_DIR}/run_vbench.py --video_dir ${I2V_OUTPUT}"
