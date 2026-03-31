#!/bin/bash
###############################################################################
# Helios VBench Run Script (env already installed)
#
# 直接激活已有环境, 下载权重, 生成 T2V + I2V
# 用法: bash helios_run.sh
###############################################################################
set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_ID="BestWishYsh/Helios-Distilled"
MODEL_DIR="${BASE_DIR}/Helios-Distilled"
VBENCH_T2V_JSON="${BASE_DIR}/VBench_aug_full_info.json"
VBENCH_I2V_JSON="${BASE_DIR}/vbench2_i2v_full_info.json"
I2V_DATA_DIR="${BASE_DIR}/datasets/vbench_i2v/data"
I2V_IMAGE_DIR="${I2V_DATA_DIR}/crop"
OUTPUT_BASE="${BASE_DIR}/outputs/vbench"
T2V_OUTPUT="${OUTPUT_BASE}/helios_t2v"
I2V_OUTPUT="${OUTPUT_BASE}/helios_i2v"
NUM_GPUS=8
NUM_SAMPLES=5

# ===================== Activate env =====================
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate helios
PY="$(which python)"
echo "Python: $($PY --version) at $PY"

# ===================== Download weights =====================
echo "[1/4] Checking weights..."
if [ -f "${MODEL_DIR}/model_index.json" ]; then
    echo "  -> Already downloaded."
else
    echo "  -> Downloading Helios-Distilled..."
    huggingface-cli download ${MODEL_ID} --local-dir "${MODEL_DIR}"
fi

# ===================== Download I2V images =====================
echo "[2/4] Checking I2V images..."
if [ -d "${I2V_IMAGE_DIR}" ] && [ "$(ls -A ${I2V_IMAGE_DIR} 2>/dev/null)" ]; then
    echo "  -> Already downloaded."
else
    echo "  -> Downloading I2V images..."
    mkdir -p "${I2V_DATA_DIR}"
    pip install -q gdown 2>/dev/null
    gdown --id 1Y_JnYnyJ3a6QhiranoX0MQVZFcTDPekZ --output "${I2V_DATA_DIR}/crop.zip"
    unzip -q "${I2V_DATA_DIR}/crop.zip" -d "${I2V_DATA_DIR}"
    rm -f "${I2V_DATA_DIR}/crop.zip"
fi

if [ -d "${I2V_IMAGE_DIR}/8-5" ]; then
    I2V_CROP_DIR="${I2V_IMAGE_DIR}/8-5"
elif [ -d "${I2V_IMAGE_DIR}/7-4" ]; then
    I2V_CROP_DIR="${I2V_IMAGE_DIR}/7-4"
else
    I2V_CROP_DIR="${I2V_IMAGE_DIR}"
fi
echo "  -> Crop dir: ${I2V_CROP_DIR}"

# ===================== T2V Generation =====================
echo "[3/4] T2V generation (${NUM_GPUS} GPUs)..."
mkdir -p "${T2V_OUTPUT}"

cat > /tmp/_helios_t2v_worker.py << 'WORKER_EOF'
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

PIDS=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=${gpu_id} $PY /tmp/_helios_t2v_worker.py \
        --worker_id ${gpu_id} --num_workers ${NUM_GPUS} \
        --model_dir "${MODEL_DIR}" --vbench_json "${VBENCH_T2V_JSON}" \
        --output_dir "${T2V_OUTPUT}" --num_samples ${NUM_SAMPLES} \
        > "${OUTPUT_BASE}/helios_t2v_w${gpu_id}.log" 2>&1 &
    PIDS+=($!)
    echo "  GPU ${gpu_id}: PID $!"
done

echo "  Waiting for T2V..."
FAIL=0
for pid in "${PIDS[@]}"; do wait $pid || FAIL=$((FAIL + 1)); done
echo "  T2V Done. $(find "${T2V_OUTPUT}" -name '*.mp4' | wc -l) videos. Failed: ${FAIL}"

# ===================== I2V Generation =====================
echo "[4/4] I2V generation (${NUM_GPUS} GPUs)..."
mkdir -p "${I2V_OUTPUT}"

cat > /tmp/_helios_i2v_worker.py << WORKER_EOF
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
            image_path = None
            for root, _, files in os.walk(I2V_CROP_DIR):
                if image_name in files:
                    image_path = os.path.join(root, image_name)
                    break
            if not image_path:
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

PIDS=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=${gpu_id} $PY /tmp/_helios_i2v_worker.py \
        --worker_id ${gpu_id} --num_workers ${NUM_GPUS} \
        --model_dir "${MODEL_DIR}" --vbench_json "${VBENCH_I2V_JSON}" \
        --output_dir "${I2V_OUTPUT}" --num_samples ${NUM_SAMPLES} \
        > "${OUTPUT_BASE}/helios_i2v_w${gpu_id}.log" 2>&1 &
    PIDS+=($!)
    echo "  GPU ${gpu_id}: PID $!"
done

echo "  Waiting for I2V..."
FAIL=0
for pid in "${PIDS[@]}"; do wait $pid || FAIL=$((FAIL + 1)); done
echo "  I2V Done. $(find "${I2V_OUTPUT}" -name '*.mp4' | wc -l) videos. Failed: ${FAIL}"

# ===================== Cleanup & Summary =====================
find "${T2V_OUTPUT}" "${I2V_OUTPUT}" -name "*.lock" -delete 2>/dev/null
rm -f /tmp/_helios_t2v_worker.py /tmp/_helios_i2v_worker.py

echo ""
echo "=============================================="
echo " Done!"
echo "=============================================="
echo "T2V: $(find "${T2V_OUTPUT}" -name '*.mp4' | wc -l) videos in ${T2V_OUTPUT}"
echo "I2V: $(find "${I2V_OUTPUT}" -name '*.mp4' | wc -l) videos in ${I2V_OUTPUT}"
echo ""
echo "Run evaluation:"
echo "  conda activate helios"
echo "  torchrun --nproc_per_node=8 run_vbench.py --video_dir ${T2V_OUTPUT}"
