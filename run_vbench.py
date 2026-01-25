"""
VBench evaluation for FastVideo, Krea, LightX2V.

GPU allocation:
- GPU 0-3: FastVideo (4 workers)
- GPU 4-5: Krea (2 workers)
- GPU 6-7: LightX2V (2 workers)

Work stealing: finished methods help others.
After generation, run VBench scoring, then upload to Enderfga/vbench.
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


def create_hf_repo():
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True)
        print(f"HF repo ready: {HF_REPO}")
    except Exception as e:
        print(f"Warning: Could not create HF repo: {e}")


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


def run_fastvideo_worker(worker_id: int, gpu_id: int, start_offset: int):
    """Run FastVideo VBench worker."""
    log_file = Path(f"{OUTPUT_BASE}/fastvideo_worker{worker_id}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    script_content = f'''
import json
import os
import sys
from pathlib import Path

def main():
    sys.path.insert(0, '/root/data/video-gen-related/FastVideo')

    from fastvideo import VideoGenerator, SamplingParam
    from fastvideo.configs.pipelines.wan import SelfForcingWan2_2_T2V480PConfig

    WORKER_ID = {worker_id}
    START_OFFSET = {start_offset}
    NUM_FRAMES = 81
    WIDTH = 832
    HEIGHT = 480
    NUM_SAMPLES_PER_PROMPT = {NUM_SAMPLES_PER_PROMPT}
    TOTAL_SAMPLES = {TOTAL_SAMPLES}
    NEGATIVE_PROMPT = '镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
    VBENCH_JSON = '/root/data/rcm/VBench_aug_full_info.json'
    OUTPUT_PATH = '/root/data/video-gen-related/outputs/vbench/fastvideo'

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Clean up stale lock files from previous interrupted runs
    import glob
    for lock_file in glob.glob(os.path.join(OUTPUT_PATH, '.*.lock')):
        os.remove(lock_file)
    print(f'[Worker {{WORKER_ID}}] Cleaned up stale lock files')

    model_id = '/root/data/video-gen-related/CausalWan2.2-I2V-A14B-Preview-Diffusers'
    pipeline_config = SelfForcingWan2_2_T2V480PConfig.from_pretrained(model_id)

    print(f'[Worker {{WORKER_ID}}] Loading FastVideo model...')
    generator = VideoGenerator.from_pretrained(
        model_path=model_id,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_precision='bf16',
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        dmd_denoising_steps=[1000, 850, 700, 550, 350, 275, 200, 125],
        pin_cpu_memory=False,
        pipeline_config=pipeline_config,
    )
    print(f'[Worker {{WORKER_ID}}] FastVideo model loaded!')

    sampling_param = SamplingParam.from_pretrained(model_id)
    sampling_param.num_frames = NUM_FRAMES
    sampling_param.width = WIDTH
    sampling_param.height = HEIGHT
    sampling_param.negative_prompt = NEGATIVE_PROMPT

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

            sampling_param.seed = seed + 1  # FastVideo doesn't support seed=0

            # Pass full .mp4 path so FastVideo uses our filename directly
            _ = generator.generate_video(prompt, output_path=str(target_path), save_video=True, sampling_param=sampling_param)
            print(f'[Worker {{WORKER_ID}}] Saved: {{target_path.name}}')
        finally:
            lock_path.unlink(missing_ok=True)

    print(f'[Worker {{WORKER_ID}}] FastVideo done!')

if __name__ == "__main__":
    main()
'''
    script_path = f"{OUTPUT_BASE}/_run_fastvideo_vbench_{worker_id}.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    cmd = f"source {CONDA_PATH}/bin/activate fastvideo && cd /root/data/video-gen-related && CUDA_VISIBLE_DEVICES={gpu_id} python {script_path}"

    print(f"[FastVideo Worker {worker_id}] Starting on GPU {gpu_id}...")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT)
    return proc


def run_krea_worker(worker_id: int, gpu_id: int, start_offset: int):
    """Run Krea VBench worker."""
    log_file = Path(f"{OUTPUT_BASE}/krea_worker{worker_id}.log")

    script_content = f'''
import json
import os
import sys
from pathlib import Path

os.chdir('/root/data/video-gen-related/realtime-video')
sys.path.insert(0, '/root/data/video-gen-related/realtime-video')

import torch

def patched_text_encoder_init(self):
    from wan.modules.tokenizers import HuggingfaceTokenizer
    from wan.modules.t5 import umt5_xxl
    from settings import MODEL_FOLDER
    torch.nn.Module.__init__(self)
    self.text_encoder = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.float32, device=torch.device('cuda')).eval().requires_grad_(False)
    pth_path = os.path.join(MODEL_FOLDER, 'Wan2.1-T2V-1.3B', 'models_t5_umt5-xxl-enc-bf16.pth')
    state_dict = torch.load(pth_path, map_location='cuda', weights_only=True)
    self.text_encoder.load_state_dict(state_dict)
    self.tokenizer = HuggingfaceTokenizer(name=os.path.join(MODEL_FOLDER, 'Wan2.1-T2V-1.3B', 'google', 'umt5-xxl/'), seq_len=512, clean='whitespace')

from utils import wan_wrapper
wan_wrapper.WanTextEncoder.__init__ = patched_text_encoder_init

from release_server import load_merge_config, load_all, GenerateParams, GenerationSession
import torchvision.io as io

torch.set_grad_enabled(False)

WORKER_ID = {worker_id}
START_OFFSET = {start_offset}
NUM_FRAMES = 81
WIDTH = 832
HEIGHT = 480
NUM_BLOCKS = 9
NUM_SAMPLES_PER_PROMPT = {NUM_SAMPLES_PER_PROMPT}
TOTAL_SAMPLES = {TOTAL_SAMPLES}
VBENCH_JSON = '/root/data/rcm/VBench_aug_full_info.json'
OUTPUT_PATH = '/root/data/video-gen-related/outputs/vbench/krea'

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Clean up stale lock files from previous interrupted runs
import glob as glob_module
for lock_file in glob_module.glob(os.path.join(OUTPUT_PATH, '.*.lock')):
    os.remove(lock_file)
print(f'[Worker {{WORKER_ID}}] Cleaned up stale lock files')

config_path = '/root/data/video-gen-related/realtime-video/configs/self_forcing_server_14b.yaml'
print(f'[Worker {{WORKER_ID}}] Loading Krea models...')
config = load_merge_config(config_path)
models = load_all(config)
print(f'[Worker {{WORKER_ID}}] Krea models loaded!')

with open(VBENCH_JSON, 'r') as f:
    vbench_data = json.load(f)

# Build sample list
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
    video_path = Path(OUTPUT_PATH) / video_name

    if video_path.exists():
        continue

    lock_path = Path(OUTPUT_PATH) / f'.{{prompt_en}}-{{idx}}.lock'
    try:
        lock_path.touch(exist_ok=False)
    except FileExistsError:
        continue

    try:
        print(f'[Worker {{WORKER_ID}}] [{{sample_idx+1}}/{{TOTAL_SAMPLES}}] {{prompt_en[:40]}}...-{{idx}}')

        params = GenerateParams(prompt=prompt, width=WIDTH, height=HEIGHT, num_blocks=NUM_BLOCKS, seed=seed, kv_cache_num_frames=3)
        all_frames = []

        def frame_callback(pixels, frame_ids, event):
            event.synchronize()
            cpu_pixels = pixels.cpu().add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
            all_frames.append(cpu_pixels)

        session = GenerationSession(params=params, config=config, frame_callback=frame_callback, models=models)

        for block_idx in range(params.num_blocks):
            session.generate_block(models)

        combined_frames = torch.cat(all_frames, dim=1)
        if combined_frames.shape[1] > NUM_FRAMES:
            combined_frames = combined_frames[:, :NUM_FRAMES]

        video_tensor = combined_frames[0].permute(0, 2, 3, 1).cpu().clamp(0, 1)
        video_tensor = (video_tensor * 255).to(torch.uint8)
        io.write_video(str(video_path), video_tensor, fps=16, video_codec='h264', options={{'crf': '18'}})
        print(f'[Worker {{WORKER_ID}}] Saved: {{video_path.name}}')

        session.dispose()
        all_frames.clear()
    finally:
        lock_path.unlink(missing_ok=True)

print(f'[Worker {{WORKER_ID}}] Krea done!')
'''
    script_path = f"{OUTPUT_BASE}/_run_krea_vbench_{worker_id}.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    cmd = f"cd /root/data/video-gen-related/realtime-video && source .venv/bin/activate && export MODEL_FOLDER='/root/data/video-gen-related' && CUDA_VISIBLE_DEVICES={gpu_id} python {script_path}"

    print(f"[Krea Worker {worker_id}] Starting on GPU {gpu_id}...")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT)
    return proc


def run_lightx2v_worker(worker_id: int, gpu_id: int, start_offset: int):
    """Run LightX2V VBench worker."""
    log_file = Path(f"{OUTPUT_BASE}/lightx2v_worker{worker_id}.log")

    script_content = f'''
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, '/root/data/video-gen-related/lightx2v')

import torch
from lightx2v.pipeline import LightX2VPipeline

torch.set_grad_enabled(False)

WORKER_ID = {worker_id}
START_OFFSET = {start_offset}
NUM_FRAMES = 81
NUM_SAMPLES_PER_PROMPT = {NUM_SAMPLES_PER_PROMPT}
TOTAL_SAMPLES = {TOTAL_SAMPLES}
NEGATIVE_PROMPT = '镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
VBENCH_JSON = '/root/data/rcm/VBench_aug_full_info.json'
OUTPUT_PATH = '/root/data/video-gen-related/outputs/vbench/lightx2v'
MODEL_PATH = '/root/data/video-gen-related/Wan2.1-T2V-14B-CausVid'
CONFIG_JSON = '/root/data/video-gen-related/lightx2v/configs/causvid/wan_t2v_causvid.json'

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Clean up stale lock files from previous interrupted runs
import glob as glob_module
for lock_file in glob_module.glob(os.path.join(OUTPUT_PATH, '.*.lock')):
    os.remove(lock_file)
print(f'[Worker {{WORKER_ID}}] Cleaned up stale lock files')

print(f'[Worker {{WORKER_ID}}] Initializing LightX2V...')
pipeline = LightX2VPipeline(task='t2v', model_path=MODEL_PATH, model_cls='wan2.1_causvid')
pipeline.create_generator(config_json=CONFIG_JSON)
print(f'[Worker {{WORKER_ID}}] Pipeline initialized!')

with open(VBENCH_JSON, 'r') as f:
    vbench_data = json.load(f)

# Build sample list
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
    save_path = Path(OUTPUT_PATH) / video_name

    if save_path.exists():
        continue

    lock_path = Path(OUTPUT_PATH) / f'.{{prompt_en}}-{{idx}}.lock'
    try:
        lock_path.touch(exist_ok=False)
    except FileExistsError:
        continue

    try:
        print(f'[Worker {{WORKER_ID}}] [{{sample_idx+1}}/{{TOTAL_SAMPLES}}] {{prompt_en[:40]}}...-{{idx}}')
        pipeline.generate(seed=seed, prompt=prompt, negative_prompt=NEGATIVE_PROMPT, save_result_path=str(save_path))
        print(f'[Worker {{WORKER_ID}}] Saved: {{save_path.name}}')
    finally:
        lock_path.unlink(missing_ok=True)

print(f'[Worker {{WORKER_ID}}] LightX2V done!')
'''
    script_path = f"{OUTPUT_BASE}/_run_lightx2v_vbench_{worker_id}.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    cmd = f"source {CONDA_PATH}/bin/activate lightx2v && cd /root/data/video-gen-related && CUDA_VISIBLE_DEVICES={gpu_id} python {script_path}"

    print(f"[LightX2V Worker {worker_id}] Starting on GPU {gpu_id}...")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT)
    return proc


def wait_for_fastvideo_loaded(worker_id: int, timeout: int = 600):
    """Wait for a FastVideo worker to finish loading model by monitoring its log."""
    log_file = Path(f"{OUTPUT_BASE}/fastvideo_worker{worker_id}.log")
    start_time = time.time()

    while time.time() - start_time < timeout:
        if log_file.exists():
            try:
                content = log_file.read_text()
                if "FastVideo model loaded!" in content:
                    print(f"[FastVideo Worker {worker_id}] Model loaded, starting next worker...")
                    return True
                if "Error" in content and "Traceback" in content:
                    print(f"[FastVideo Worker {worker_id}] Error detected in log")
                    return False
            except:
                pass
        time.sleep(5)

    print(f"[FastVideo Worker {worker_id}] Timeout waiting for model load")
    return False


def run_vbench_eval(method_name: str, video_dir: str):
    """Run VBench evaluation on generated videos."""
    print(f"\n[{method_name}] Running VBench evaluation...")

    # Count videos first
    video_count = len([f for f in os.listdir(video_dir) if f.endswith('.mp4')]) if os.path.exists(video_dir) else 0
    print(f"[{method_name}] Found {video_count} videos to evaluate")

    if video_count == 0:
        print(f"[{method_name}] No videos found, skipping evaluation")
        return False

    # Run VBench evaluation with rcm conda env
    cmd = f"""
    source {CONDA_PATH}/bin/activate rcm && \
    cd /root/data/FAR-World && \
    torchrun --nproc_per_node=8 run_vbench.py --video_dir {video_dir}
    """

    result = subprocess.run(["bash", "-c", cmd], capture_output=False)

    if result.returncode == 0:
        print(f"[{method_name}] VBench evaluation complete!")
        return True
    else:
        print(f"[{method_name}] VBench evaluation failed!")
        return False


def main():
    print("=" * 60)
    print("VBench Evaluation - FastVideo, Krea, LightX2V")
    print("  FastVideo (8-step)  -> GPU 0-3 (4 workers)")
    print("  Krea (4-step)       -> GPU 4-5 (2 workers)")
    print("  LightX2V (9-step)   -> GPU 6-7 (2 workers)")
    print(f"  Prompts: {NUM_PROMPTS}, Samples/prompt: {NUM_SAMPLES_PER_PROMPT}")
    print(f"  Total samples: {TOTAL_SAMPLES}")
    print(f"  Upload to: {HF_REPO}")
    print("=" * 60)

    create_hf_repo()
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    processes = {}

    # Start FastVideo workers (GPU 0-3)
    fastvideo_procs = []
    n_fastvideo = 4
    for i in range(n_fastvideo):
        offset = (i * TOTAL_SAMPLES) // n_fastvideo
        proc = run_fastvideo_worker(i, gpu_id=i, start_offset=offset)
        fastvideo_procs.append(proc)
    processes["fastvideo"] = {
        "procs": fastvideo_procs,
        "output_dir": f"{OUTPUT_BASE}/fastvideo",
        "done": False,
    }

    # Start Krea workers (GPU 4-5)
    krea_procs = []
    n_krea = 2
    for i in range(n_krea):
        offset = (i * TOTAL_SAMPLES) // n_krea
        proc = run_krea_worker(i, gpu_id=4 + i, start_offset=offset)
        krea_procs.append(proc)
    processes["krea"] = {
        "procs": krea_procs,
        "output_dir": f"{OUTPUT_BASE}/krea",
        "done": False,
    }

    # Start LightX2V workers (GPU 6-7)
    lightx2v_procs = []
    n_lightx2v = 2
    for i in range(n_lightx2v):
        offset = (i * TOTAL_SAMPLES) // n_lightx2v
        proc = run_lightx2v_worker(i, gpu_id=6 + i, start_offset=offset)
        lightx2v_procs.append(proc)
    processes["lightx2v"] = {
        "procs": lightx2v_procs,
        "output_dir": f"{OUTPUT_BASE}/lightx2v",
        "done": False,
    }

    print("\nAll processes started. Monitoring for completion...")

    # Track work stealing
    helper_procs = {name: [] for name in processes}
    work_stealing_done = {name: False for name in processes}

    # Phase 1: Wait for all generation to complete
    while any(not p["done"] for p in processes.values()):
        for name, info in processes.items():
            if info["done"]:
                continue

            all_procs = info["procs"] + helper_procs[name]
            done = all(p.poll() is not None for p in all_procs)

            if done:
                success = all(p.returncode == 0 for p in all_procs)
                if success:
                    print(f"\n[{name}] Generation complete!")
                else:
                    print(f"\n[{name}] Generation failed! Check logs.")
                info["done"] = True

        # Work stealing: spawn helpers for incomplete methods
        for src_name, src_info in processes.items():
            if not src_info["done"] or work_stealing_done[src_name]:
                continue

            # Find methods that are still running
            for dst_name, dst_info in processes.items():
                if dst_info["done"]:
                    continue

                # Determine which GPUs to use for helping
                if src_name == "fastvideo":
                    gpus_to_use = [0, 1, 2, 3]
                elif src_name == "krea":
                    gpus_to_use = [4, 5]
                else:  # lightx2v
                    gpus_to_use = [6, 7]

                print(f"\n[Work Stealing] {src_name} done, helping {dst_name} with GPUs {gpus_to_use}...")
                for i, gpu_id in enumerate(gpus_to_use):
                    worker_id = 20 + gpu_id
                    offset = (gpu_id * TOTAL_SAMPLES) // 8
                    if dst_name == "fastvideo":
                        proc = run_fastvideo_worker(worker_id, gpu_id=gpu_id, start_offset=offset)
                    elif dst_name == "krea":
                        proc = run_krea_worker(worker_id, gpu_id=gpu_id, start_offset=offset)
                    else:
                        proc = run_lightx2v_worker(worker_id, gpu_id=gpu_id, start_offset=offset)
                    helper_procs[dst_name].append(proc)

                work_stealing_done[src_name] = True
                break  # Only help one method at a time

        time.sleep(10)

    print("\n" + "=" * 60)
    print("Phase 1 Complete: All generation done!")
    print("Phase 2: Running VBench evaluation and uploading...")
    print("=" * 60)

    # Phase 2: Score and upload each method sequentially (VBench uses all 8 GPUs)
    for name, info in processes.items():
        print(f"\n[{name}] Running VBench evaluation...")
        run_vbench_eval(name, info["output_dir"])

        print(f"[{name}] Uploading results...")
        upload_results(name, info["output_dir"])

    print("\n" + "=" * 60)
    print("All tasks completed!")
    print(f"Results: https://huggingface.co/datasets/{HF_REPO}")
    print("=" * 60)


if __name__ == "__main__":
    main()
