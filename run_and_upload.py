"""
Run Krea + FastVideo on 8 GPUs with work stealing.

GPU allocation:
- GPU 0: Krea (1 worker)
- GPU 1-7: FastVideo (7 workers)
- When Krea finishes, GPU 0 joins FastVideo

All workers try all prompts but skip existing files = automatic work stealing.
Results uploaded to: Enderfga/related
"""
import subprocess
import os
import json
import time
from pathlib import Path

CONDA_PATH = "/root/FGA/miniconda3"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "Enderfga/related"
PROMPT_FILE = "/root/data/video-gen-related/eval_caption_t2v_paper.json"
OUTPUT_BASE = "/root/data/video-gen-related/outputs"

with open(PROMPT_FILE, "r") as f:
    ALL_PROMPTS = json.load(f)
NUM_PROMPTS = len(ALL_PROMPTS)


def create_hf_repo():
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True)
        print(f"HF repo ready: {HF_REPO}")
    except Exception as e:
        print(f"Warning: Could not create HF repo: {e}")


def upload_results(method_name: str, output_dir: str):
    import zipfile

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
    """Run FastVideo worker - processes ALL prompts but starts from different offset for load balancing."""
    log_file = Path(f"{OUTPUT_BASE}/fastvideo_worker{worker_id}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    script_content = f'''
import json
import os
import sys
import glob
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
    SEEDS = [(42, 'seed_0'), (1, 'seed_1')]  # (actual_seed, dir_name), 42 uploaded as seed_0
    NUM_PROMPTS = {NUM_PROMPTS}
    NEGATIVE_PROMPT = '镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
    PROMPT_FILE = '/root/data/video-gen-related/eval_caption_t2v_paper.json'
    OUTPUT_PATH = '/root/data/video-gen-related/outputs/fastvideo_8step'

    os.makedirs(OUTPUT_PATH, exist_ok=True)

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

    with open(PROMPT_FILE, 'r') as f:
        prompts_data = json.load(f)

    # Rotated indices for load balancing
    all_indices = list(range(NUM_PROMPTS))
    rotated_indices = all_indices[START_OFFSET:] + all_indices[:START_OFFSET]

    for actual_seed, dir_name in SEEDS:
        sampling_param.seed = actual_seed
        seed_dir = Path(OUTPUT_PATH) / dir_name
        seed_dir.mkdir(parents=True, exist_ok=True)

        for idx in rotated_indices:
            item = prompts_data[idx]
            prompt = item['prompt']
            target_path = seed_dir / f'video_{{idx:03d}}.mp4'

            # Resume: skip if target file exists
            if target_path.exists():
                continue

            # Lock file for coordination
            lock_path = seed_dir / f'.video_{{idx:03d}}.lock'
            try:
                lock_path.touch(exist_ok=False)
            except FileExistsError:
                continue

            try:
                print(f'[Worker {{WORKER_ID}}] [{{dir_name}}] [{{idx+1}}/{{NUM_PROMPTS}}] {{prompt[:50]}}...')

                # Use temp directory to avoid conflicts with other workers
                import tempfile
                with tempfile.TemporaryDirectory() as tmp_dir:
                    _ = generator.generate_video(prompt, output_path=tmp_dir, save_video=True, sampling_param=sampling_param)

                    # Find generated file in temp dir
                    tmp_files = glob.glob(os.path.join(tmp_dir, '*.mp4'))
                    if tmp_files:
                        import shutil
                        shutil.move(tmp_files[0], str(target_path))
                        print(f'[Worker {{WORKER_ID}}] Saved: {{target_path}}')
            finally:
                lock_path.unlink(missing_ok=True)

    print(f'[Worker {{WORKER_ID}}] FastVideo done!')

if __name__ == "__main__":
    main()
'''
    script_path = f"{OUTPUT_BASE}/_run_fastvideo_{worker_id}.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    cmd = f"source {CONDA_PATH}/bin/activate fastvideo && cd /root/data/video-gen-related && CUDA_VISIBLE_DEVICES={gpu_id} python {script_path}"

    print(f"[FastVideo Worker {worker_id}] Starting on GPU {gpu_id}...")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT)
    return proc


def run_krea_worker(worker_id: int, gpu_id: int, start_offset: int):
    """Run Krea worker - processes ALL prompts but starts from different offset for load balancing."""
    log_file = Path(f"{OUTPUT_BASE}/krea_worker{worker_id}.log")

    script_content = f'''
import json
import os
import sys
import time
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
SEEDS = [0, 1]
NUM_BLOCKS = 9
NUM_PROMPTS = {NUM_PROMPTS}
PROMPT_FILE = '/root/data/video-gen-related/eval_caption_t2v_paper.json'
OUTPUT_PATH = '/root/data/video-gen-related/outputs/krea_4step'

config_path = '/root/data/video-gen-related/realtime-video/configs/self_forcing_server_14b.yaml'
print(f'[Worker {{WORKER_ID}}] Loading Krea models...')
config = load_merge_config(config_path)
models = load_all(config)
print(f'[Worker {{WORKER_ID}}] Krea models loaded!')

with open(PROMPT_FILE, 'r') as f:
    prompts_data = json.load(f)

# Create rotated index list starting from offset for load balancing
all_indices = list(range(NUM_PROMPTS))
rotated_indices = all_indices[START_OFFSET:] + all_indices[:START_OFFSET]

for seed in SEEDS:
    seed_dir = Path(OUTPUT_PATH) / f'seed_{{seed}}'
    seed_dir.mkdir(parents=True, exist_ok=True)

    for idx in rotated_indices:
        item = prompts_data[idx]
        prompt = item['prompt']
        video_path = seed_dir / f'video_{{idx:03d}}.mp4'

        if video_path.exists():
            continue  # Skip silently - another worker did it

        # Double check with lock file
        lock_path = seed_dir / f'.video_{{idx:03d}}.lock'
        try:
            lock_path.touch(exist_ok=False)
        except FileExistsError:
            continue  # Another worker is processing

        try:
            print(f'[Worker {{WORKER_ID}}] [Seed {{seed}}] [{{idx+1}}/{{NUM_PROMPTS}}] {{prompt[:50]}}...')

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
            print(f'[Worker {{WORKER_ID}}] Saved: {{video_path}}')

            session.dispose()
            all_frames.clear()
        finally:
            lock_path.unlink(missing_ok=True)

print(f'[Worker {{WORKER_ID}}] Krea done!')
'''
    script_path = f"{OUTPUT_BASE}/_run_krea_{worker_id}.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    cmd = f"cd /root/data/video-gen-related/realtime-video && source .venv/bin/activate && export MODEL_FOLDER='/root/data/video-gen-related' && CUDA_VISIBLE_DEVICES={gpu_id} python {script_path}"

    print(f"[Krea Worker {worker_id}] Starting on GPU {gpu_id}...")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT)
    return proc


def run_lightx2v_worker(worker_id: int, gpu_id: int, start_offset: int):
    """Run LightX2V worker - processes ALL prompts but starts from different offset."""
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
SEEDS = [0, 1]
NUM_PROMPTS = {NUM_PROMPTS}
NEGATIVE_PROMPT = '镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
PROMPT_FILE = '/root/data/video-gen-related/eval_caption_t2v_paper.json'
OUTPUT_PATH = '/root/data/video-gen-related/outputs/lightx2v_9step'
MODEL_PATH = '/root/data/video-gen-related/Wan2.1-T2V-14B-CausVid'
CONFIG_JSON = '/root/data/video-gen-related/lightx2v/configs/causvid/wan_t2v_causvid.json'

os.makedirs(OUTPUT_PATH, exist_ok=True)

print(f'[Worker {{WORKER_ID}}] Initializing LightX2V...')
pipeline = LightX2VPipeline(task='t2v', model_path=MODEL_PATH, model_cls='wan2.1_causvid')
pipeline.create_generator(config_json=CONFIG_JSON)
print(f'[Worker {{WORKER_ID}}] Pipeline initialized!')

with open(PROMPT_FILE, 'r') as f:
    prompts_data = json.load(f)

# Rotated indices for load balancing
all_indices = list(range(NUM_PROMPTS))
rotated_indices = all_indices[START_OFFSET:] + all_indices[:START_OFFSET]

for seed in SEEDS:
    seed_dir = Path(OUTPUT_PATH) / f'seed_{{seed}}'
    seed_dir.mkdir(parents=True, exist_ok=True)

    for idx in rotated_indices:
        item = prompts_data[idx]
        prompt = item['prompt']
        save_path = seed_dir / f'video_{{idx:03d}}.mp4'

        if save_path.exists():
            continue  # Skip - another worker did it

        # Lock file for coordination
        lock_path = seed_dir / f'.video_{{idx:03d}}.lock'
        try:
            lock_path.touch(exist_ok=False)
        except FileExistsError:
            continue

        try:
            print(f'[Worker {{WORKER_ID}}] [Seed {{seed}}] [{{idx+1}}/{{NUM_PROMPTS}}] {{prompt[:50]}}...')
            pipeline.generate(seed=seed, prompt=prompt, negative_prompt=NEGATIVE_PROMPT, save_result_path=str(save_path))
            print(f'[Worker {{WORKER_ID}}] Saved: {{save_path}}')
        finally:
            lock_path.unlink(missing_ok=True)

print(f'[Worker {{WORKER_ID}}] LightX2V done!')
'''
    script_path = f"{OUTPUT_BASE}/_run_lightx2v_{worker_id}.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    cmd = f"source {CONDA_PATH}/bin/activate lightx2v && cd /root/data/video-gen-related && CUDA_VISIBLE_DEVICES={gpu_id} python {script_path}"

    print(f"[LightX2V Worker {worker_id}] Starting on GPU {gpu_id}...")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT)
    return proc


def wait_for_model_loaded(worker_id: int, timeout: int = 600):
    """Wait for a FastVideo worker to finish loading model by monitoring its log."""
    log_file = Path(f"{OUTPUT_BASE}/fastvideo_worker{worker_id}.log")
    start_time = time.time()

    while time.time() - start_time < timeout:
        if log_file.exists():
            try:
                content = log_file.read_text()
                if "FastVideo model loaded!" in content:
                    print(f"[Worker {worker_id}] Model loaded, starting next worker...")
                    return True
                if "Error" in content or "Exception" in content:
                    print(f"[Worker {worker_id}] Error detected in log")
                    return False
            except:
                pass
        time.sleep(5)

    print(f"[Worker {worker_id}] Timeout waiting for model load")
    return False


def main():
    print("=" * 60)
    print("Video Generation - FastVideo on 8 GPUs")
    print("  FastVideo (8-step)  -> GPU 0-7 (8 workers)")
    print(f"  Prompts: {NUM_PROMPTS}, Seeds: 2")
    print(f"  Upload to: {HF_REPO}")
    print("=" * 60)

    create_hf_repo()
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    processes = {}

    # Start FastVideo workers
    fastvideo_procs = []
    n_fastvideo = 8
    for i in range(n_fastvideo):
        offset = (i * NUM_PROMPTS) // n_fastvideo
        proc = run_fastvideo_worker(i, gpu_id=i, start_offset=offset)
        fastvideo_procs.append(proc)

    processes["fastvideo"] = {
        "procs": fastvideo_procs,
        "output_dir": f"{OUTPUT_BASE}/fastvideo_8step",
        "uploaded": False,
    }

    print("\nAll workers started. Monitoring for completion...")

    # Track work stealing state
    krea_done = False
    fastvideo_helper_spawned = False
    fastvideo_helper_procs = []

    while any(not p["uploaded"] for p in processes.values()):
        for name, info in processes.items():
            if info["uploaded"]:
                continue

            # For fastvideo, also check helper procs
            base_procs = info["procs"]
            if name == "fastvideo":
                all_procs = base_procs + fastvideo_helper_procs
            else:
                all_procs = base_procs
            done = all(p.poll() is not None for p in all_procs)
            success = done and all(p.returncode == 0 for p in all_procs)

            if done:
                if success:
                    print(f"\n[{name}] Generation complete!")
                    upload_results(name, info["output_dir"])
                else:
                    print(f"\n[{name}] Failed! Check logs at {OUTPUT_BASE}/{name}*.log")
                info["uploaded"] = True

                # Mark krea as done for work stealing
                if name == "krea":
                    krea_done = True

        # Work stealing: when Krea finishes, spawn FastVideo helper on GPU 0
        if krea_done and not fastvideo_helper_spawned and not processes["fastvideo"]["uploaded"]:
            print("\n[Work Stealing] Krea done, spawning FastVideo helper on GPU 0...")
            # Use worker ID 10 to avoid log file conflicts
            proc = run_fastvideo_worker(10, gpu_id=0, start_offset=NUM_PROMPTS // 2)
            fastvideo_helper_procs.append(proc)
            fastvideo_helper_spawned = True

        time.sleep(10)

    print("\n" + "=" * 60)
    print("All tasks completed!")
    print(f"Results: https://huggingface.co/datasets/{HF_REPO}")
    print("=" * 60)


if __name__ == "__main__":
    main()
