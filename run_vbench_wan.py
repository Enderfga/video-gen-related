"""
VBench evaluation for Wan2.1 (diffusers pipeline).

8 GPU workers for parallel generation.
"""
import subprocess
import os
import json
import time
import zipfile
from pathlib import Path

CONDA_PATH = "/root/FGA/miniconda3"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "Enderfga/vbench"
VBENCH_JSON = "/root/data/rcm/VBench_aug_full_info.json"
OUTPUT_BASE = "/root/data/video-gen-related/outputs/vbench"
NUM_SAMPLES_PER_PROMPT = 5

# Load VBench prompts
with open(VBENCH_JSON, "r") as f:
    VBENCH_DATA = json.load(f)
NUM_PROMPTS = len(VBENCH_DATA)
TOTAL_SAMPLES = NUM_PROMPTS * NUM_SAMPLES_PER_PROMPT


def upload_results(method_name: str, output_dir: str):
    if not os.path.exists(output_dir):
        print(f"[{method_name}] No output directory found, skipping upload")
        return False

    zip_name = f"{method_name}.zip"
    zip_path = f"{OUTPUT_BASE}/{zip_name}"

    print(f"[{method_name}] Creating zip...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zf.write(file_path, arcname)

    print(f"[{method_name}] Uploading to HuggingFace...")
    result = subprocess.run(
        ["huggingface-cli", "upload", "--repo-type", "dataset", "--token", HF_TOKEN,
         HF_REPO, zip_path, zip_name],
        capture_output=True, text=True,
    )

    if result.returncode == 0:
        print(f"[{method_name}] Upload complete!")
        os.remove(zip_path)
        return True
    else:
        print(f"[{method_name}] Upload failed: {result.stderr}")
        return False


def run_wan_worker(worker_id: int, gpu_id: int, start_offset: int):
    """Run Wan2.1 VBench worker."""
    log_file = Path(f"{OUTPUT_BASE}/wan_worker{worker_id}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    script_content = f'''
import json
import os
import torch
from pathlib import Path
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

WORKER_ID = {worker_id}
START_OFFSET = {start_offset}
NUM_FRAMES = 81
WIDTH = 832
HEIGHT = 480
NUM_SAMPLES_PER_PROMPT = {NUM_SAMPLES_PER_PROMPT}
TOTAL_SAMPLES = {TOTAL_SAMPLES}
NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
VBENCH_JSON = '/root/data/rcm/VBench_aug_full_info.json'
OUTPUT_PATH = '/root/data/video-gen-related/outputs/vbench/wan'

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Clean up stale lock files from previous interrupted runs
import glob
for lock_file in glob.glob(os.path.join(OUTPUT_PATH, '.*.lock')):
    os.remove(lock_file)
print(f'[Worker {{WORKER_ID}}] Cleaned up stale lock files')

print(f'[Worker {{WORKER_ID}}] Loading Wan2.1 model...')
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
flow_shift = 5.0
scheduler = UniPCMultistepScheduler(
    prediction_type='flow_prediction',
    use_flow_sigmas=True,
    num_train_timesteps=1000,
    flow_shift=flow_shift
)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.scheduler = scheduler
pipe.to("cuda")
print(f'[Worker {{WORKER_ID}}] Wan2.1 model loaded!')

with open(VBENCH_JSON, 'r') as f:
    vbench_data = json.load(f)

# Build sample list: (prompt, prompt_en, seed, sample_idx)
samples = []
for prompt_item in vbench_data:
    prompt = prompt_item.get('aug_prompt_en', prompt_item.get('prompt', ''))
    prompt_en = prompt_item.get('prompt_en', prompt_item.get('prompt', ''))
    for idx in range(NUM_SAMPLES_PER_PROMPT):
        samples.append((prompt, prompt_en, len(samples), idx))

# Rotated indices for load balancing
all_indices = list(range(TOTAL_SAMPLES))
rotated_indices = all_indices[START_OFFSET:] + all_indices[:START_OFFSET]

for sample_idx in rotated_indices:
    prompt, prompt_en, seed, idx = samples[sample_idx]
    video_name = f'{{prompt_en}}-{{idx}}.mp4'
    target_path = Path(OUTPUT_PATH) / video_name

    if target_path.exists():
        continue

    lock_path = Path(OUTPUT_PATH) / f'.{{prompt_en}}-{{idx}}.lock'
    try:
        lock_path.touch(exist_ok=False)
    except FileExistsError:
        continue

    try:
        print(f'[Worker {{WORKER_ID}}] [{{sample_idx+1}}/{{TOTAL_SAMPLES}}] {{prompt_en[:40]}}...-{{idx}}')

        # Generate video
        generator = torch.Generator(device="cuda").manual_seed(seed)
        output = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            guidance_scale=6.0,
            generator=generator,
        ).frames[0]

        export_to_video(output, str(target_path), fps=16)
        print(f'[Worker {{WORKER_ID}}] Saved: {{target_path.name}}')
    except Exception as e:
        print(f'[Worker {{WORKER_ID}}] Error: {{e}}')
    finally:
        lock_path.unlink(missing_ok=True)

print(f'[Worker {{WORKER_ID}}] Wan2.1 done!')
'''
    script_path = f"{OUTPUT_BASE}/_run_wan_vbench_{worker_id}.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    cmd = f"source {CONDA_PATH}/bin/activate rcm && CUDA_VISIBLE_DEVICES={gpu_id} python {script_path}"

    print(f"[Wan Worker {worker_id}] Starting on GPU {gpu_id}...")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT)
    return proc


def count_videos(output_dir: str) -> int:
    """Count completed videos in output directory."""
    if not os.path.exists(output_dir):
        return 0
    return len([f for f in os.listdir(output_dir) if f.endswith('.mp4')])


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    wan_output = f"{OUTPUT_BASE}/wan"

    print("=" * 60)
    print("VBench Evaluation - Wan2.1")
    print(f"Total prompts: {NUM_PROMPTS}")
    print(f"Samples per prompt: {NUM_SAMPLES_PER_PROMPT}")
    print(f"Total videos to generate: {TOTAL_SAMPLES}")
    print("=" * 60)

    # Phase 1: Generation with 8 workers
    print("\n[Phase 1] Starting Wan2.1 generation with 8 workers...")

    wan_procs = []
    for i in range(8):
        offset = i * (TOTAL_SAMPLES // 8)
        proc = run_wan_worker(i, i, offset)
        wan_procs.append(proc)

    # Monitor progress
    while True:
        wan_done = all(p.poll() is not None for p in wan_procs)
        wan_count = count_videos(wan_output)

        print(f"\r[Progress] Wan: {wan_count}/{TOTAL_SAMPLES}", end="", flush=True)

        if wan_done:
            print(f"\n[Phase 1] Generation complete! Total: {wan_count}/{TOTAL_SAMPLES}")
            break

        time.sleep(10)

    # Phase 2: Upload
    print("\n[Phase 2] Uploading results...")
    upload_results("wan", wan_output)

    print("\n" + "=" * 60)
    print("VBench Wan2.1 evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
