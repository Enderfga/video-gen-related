"""
VBench I2V evaluation for FastVideo.

8 GPU workers for parallel generation.
Uses vbench2_i2v_aug_full_info.json for prompts and source images.

Optimized for H200 141GB: BF16, full GPU, no CPU offload.

Multi-machine support:
  Machine 0 (first half):  python run_vbench_i2v.py --machine 0
  Machine 1 (second half): python run_vbench_i2v.py --machine 1
"""
import subprocess
import os
import json
import time
import zipfile
import argparse
from pathlib import Path

CONDA_PATH = "/root/FGA/miniconda3"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "Enderfga/vbench"
VBENCH_I2V_JSON = "/root/data/FAR-World/assets/data/meta/vbench/vbench2_i2v_aug_full_info.json"
IMAGE_DIR = "/root/data/FAR-World/vbench2_beta_i2v/data/crop/7-4"
OUTPUT_BASE = "/root/data/video-gen-related/outputs/vbench"
EMBEDDINGS_DIR = f"{OUTPUT_BASE}/embeddings"
EMBEDDINGS_PATH = f"{EMBEDDINGS_DIR}/embeddings.pt"
NUM_SAMPLES_PER_PROMPT = 5

# Load VBench I2V prompts
with open(VBENCH_I2V_JSON, "r") as f:
    VBENCH_DATA = json.load(f)
NUM_PROMPTS = len(VBENCH_DATA)
TOTAL_SAMPLES = NUM_PROMPTS * NUM_SAMPLES_PER_PROMPT


def precompute_embeddings():
    """Precompute text embeddings before running workers."""
    done_flag = f"{EMBEDDINGS_DIR}/.done"
    if os.path.exists(done_flag):
        print("[Precompute] Embeddings already exist, skipping precomputation.")
        return True

    print("[Precompute] Computing text embeddings...")
    log_file = f"{OUTPUT_BASE}/precompute_embeddings.log"
    cmd = f"source {CONDA_PATH}/bin/activate fastvideo && cd /root/data/video-gen-related && python precompute_embeddings.py"

    with open(log_file, "w") as f:
        proc = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT)

    proc.wait()

    if os.path.exists(done_flag):
        print("[Precompute] Embeddings computed successfully!")
        return True
    else:
        print(f"[Precompute] Failed! Check log: {log_file}")
        return False


def upload_results(method_name: str, output_dir: str):
    if not os.path.exists(output_dir):
        print(f"[{method_name}] No output directory found, skipping upload")
        return False

    zip_name = f"{method_name}.zip"
    zip_path = f"{OUTPUT_BASE}/{zip_name}"

    print(f"[{method_name}] Creating zip...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
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


def run_fastvideo_i2v_worker(worker_id: int, gpu_id: int, start_offset: int, samples_for_machine: int = TOTAL_SAMPLES):
    """Run FastVideo I2V VBench worker."""
    log_file = Path(f"{OUTPUT_BASE}/fastvideo_i2v_worker{worker_id}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Each worker handles 1/8 of the samples assigned to this machine
    samples_per_worker = samples_for_machine // 8

    script_content = f'''
import json
import os
import sys
import glob
from pathlib import Path

sys.path.insert(0, '/root/data/video-gen-related/FastVideo')

from fastvideo import VideoGenerator, SamplingParam

def main():
    WORKER_ID = {worker_id}
    START_OFFSET = {start_offset}
    NUM_FRAMES = 81
    WIDTH = 832
    HEIGHT = 480
    NUM_SAMPLES_PER_PROMPT = {NUM_SAMPLES_PER_PROMPT}
    SAMPLES_PER_WORKER = {samples_per_worker}
    NEGATIVE_PROMPT = '镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
    VBENCH_I2V_JSON = '/root/data/FAR-World/assets/data/meta/vbench/vbench2_i2v_aug_full_info.json'
    IMAGE_DIR = '/root/data/FAR-World/vbench2_beta_i2v/data/crop/7-4'
    OUTPUT_PATH = '/root/data/video-gen-related/outputs/vbench/fastvideo_i2v'

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Clean up stale lock files from previous interrupted runs (only worker 0)
    if WORKER_ID == 0:
        for lock_file in glob.glob(os.path.join(OUTPUT_PATH, '.*.lock')):
            try:
                os.remove(lock_file)
            except FileNotFoundError:
                pass
        print(f'[Worker {{WORKER_ID}}] Cleaned up stale lock files')

    model_id = 'FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers'

    print(f'[Worker {{WORKER_ID}}] Loading FastVideo I2V model...')
    generator = VideoGenerator.from_pretrained(
        model_id,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,     # Full GPU
        dit_precision='bf16',      # BF16 to fit in 141GB
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        dmd_denoising_steps=[1000, 850, 700, 550, 350, 275, 200, 125],
        pin_cpu_memory=False,
    )
    print(f'[Worker {{WORKER_ID}}] FastVideo I2V model loaded!')

    sampling_param = SamplingParam.from_pretrained(model_id)
    sampling_param.num_frames = NUM_FRAMES
    sampling_param.width = WIDTH
    sampling_param.height = HEIGHT
    sampling_param.negative_prompt = NEGATIVE_PROMPT

    with open(VBENCH_I2V_JSON, 'r') as f:
        vbench_data = json.load(f)

    # Build sample list: (aug_prompt, prompt_en, image_name, seed, sample_idx)
    samples = []
    for prompt_item in vbench_data:
        aug_prompt = prompt_item.get('aug_prompt_en', prompt_item.get('prompt_en', ''))
        prompt_en = prompt_item.get('prompt_en', '')
        image_name = prompt_item.get('image_name', '')
        for idx in range(NUM_SAMPLES_PER_PROMPT):
            samples.append((aug_prompt, prompt_en, image_name, len(samples), idx))

    # This worker handles samples [START_OFFSET, START_OFFSET + SAMPLES_PER_WORKER)
    end_offset = START_OFFSET + SAMPLES_PER_WORKER
    my_indices = list(range(START_OFFSET, end_offset))

    for i, sample_idx in enumerate(my_indices):
        aug_prompt, prompt_en, image_name, seed, idx = samples[sample_idx]
        video_name = f'{{prompt_en}}-{{idx}}.mp4'
        target_path = Path(OUTPUT_PATH) / video_name

        if target_path.exists():
            continue

        # Check if source image exists
        image_path = os.path.join(IMAGE_DIR, image_name)
        if not os.path.exists(image_path):
            print(f'[Worker {{WORKER_ID}}] Image not found: {{image_name}}, skipping')
            continue

        try:
            print(f'[Worker {{WORKER_ID}}] [{{i+1}}/{{SAMPLES_PER_WORKER}}] {{prompt_en[:40]}}...-{{idx}}')

            sampling_param.seed = seed + 1  # FastVideo doesn't support seed=0

            # Generate I2V video with image_path
            _ = generator.generate_video(
                aug_prompt,
                image_path=image_path,
                output_path=str(target_path),
                save_video=True,
                sampling_param=sampling_param
            )
            print(f'[Worker {{WORKER_ID}}] Saved: {{target_path.name}}')
        except Exception as e:
            print(f'[Worker {{WORKER_ID}}] Error: {{e}}')

    print(f'[Worker {{WORKER_ID}}] FastVideo I2V done!')

if __name__ == '__main__':
    main()
'''
    script_path = f"{OUTPUT_BASE}/_run_fastvideo_i2v_vbench_{worker_id}.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    # gpu_id can be "0,1" for multi-GPU workers
    cmd = f"source {CONDA_PATH}/bin/activate fastvideo && cd /root/data/video-gen-related && CUDA_VISIBLE_DEVICES={gpu_id} python {script_path}"

    print(f"[FastVideo I2V Worker {worker_id}] Starting on GPU {gpu_id}...")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT)
    return proc


def wait_for_model_loaded(worker_id: int, timeout: int = 600):
    """Wait for worker to finish loading model."""
    log_file = Path(f"{OUTPUT_BASE}/fastvideo_i2v_worker{worker_id}.log")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if log_file.exists():
            try:
                content = log_file.read_text()
                if "FastVideo I2V model loaded!" in content:
                    print(f"[Worker {worker_id}] Model loaded!")
                    return True
                if "Error" in content and "Traceback" in content:
                    print(f"[Worker {worker_id}] Error detected")
                    return False
            except:
                pass
        time.sleep(5)
    print(f"[Worker {worker_id}] Timeout waiting for model load")
    return False


def count_videos(output_dir: str) -> int:
    """Count completed videos in output directory."""
    if not os.path.exists(output_dir):
        return 0
    return len([f for f in os.listdir(output_dir) if f.endswith('.mp4')])


def run_vbench_i2v_evaluation(video_dir: str):
    """Run VBench I2V evaluation on generated videos."""
    eval_script = f'''
import os
import json
import sys

os.environ['VBENCH_CACHE_DIR'] = '/root/data/FAR-World/experiments/pretrained_models/vbench'
sys.path.insert(0, '/root/data/FAR-World')

from vbench2_beta_i2v import VBenchI2V

DIMENSIONS = [
    'camera_motion', 'i2v_subject', 'i2v_background',
    'subject_consistency', 'motion_smoothness', 'background_consistency',
    'dynamic_degree', 'aesthetic_quality', 'imaging_quality'
]

METRICS_NORMALIZATION_RANGES = {{
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
}}

def norm(metric, key):
    range_ = METRICS_NORMALIZATION_RANGES.get(key, [0.0, 1.0])
    metric = max(metric, range_[0])
    metric = min(metric, range_[1])
    metric = (metric - range_[0]) / (range_[1] - range_[0])
    return metric

video_dir = '{video_dir}'
save_dir = '{video_dir}/vbench_info'
os.makedirs(save_dir, exist_ok=True)

print('Initializing VBench I2V evaluator...')
evaluator = VBenchI2V(
    device='cuda:0',
    full_json_dir='/root/data/FAR-World/assets/data/meta/vbench/vbench2_i2v_full_info.json',
    output_path=save_dir
)

eval_info_dict = {{}}

for metric_dimension in DIMENSIONS:
    vbench_info_path = os.path.join(save_dir, f'{{metric_dimension}}_eval_results.json')
    if os.path.exists(vbench_info_path):
        print(f'Loading cached {{metric_dimension}} results...')
        with open(vbench_info_path, 'r') as fr:
            metric_dict = json.load(fr)
            eval_info_dict[metric_dimension] = metric_dict[metric_dimension][0]
    else:
        print(f'Evaluating {{metric_dimension}}...')
        result = evaluator.evaluate(
            videos_path=video_dir,
            name=metric_dimension,
            dimension_list=[metric_dimension],
            resolution='480p',
            local=True,
        )
        eval_info_dict.update(result)

# Calculate composite scores
eval_info_dict['quality_score'] = (
    norm(eval_info_dict.get('subject_consistency', 0), 'subject_consistency') +
    norm(eval_info_dict.get('background_consistency', 0), 'background_consistency') +
    norm(eval_info_dict.get('motion_smoothness', 0), 'motion_smoothness') +
    norm(eval_info_dict.get('dynamic_degree', 0), 'dynamic_degree') * 0.5 +
    norm(eval_info_dict.get('aesthetic_quality', 0), 'aesthetic_quality') +
    norm(eval_info_dict.get('imaging_quality', 0), 'imaging_quality')
) / 5.5

eval_info_dict['i2v_score'] = (
    norm(eval_info_dict.get('i2v_subject', 0), 'i2v_subject') +
    norm(eval_info_dict.get('i2v_background', 0), 'i2v_background') +
    norm(eval_info_dict.get('camera_motion', 0), 'camera_motion') * 0.1
) / 2.1

eval_info_dict['overall_score'] = 0.5 * eval_info_dict['i2v_score'] + 0.5 * eval_info_dict['quality_score']

# Save final results
results_path = os.path.join(save_dir, 'final_results.json')
with open(results_path, 'w') as f:
    json.dump(eval_info_dict, f, indent=2)

print('\\n' + '=' * 60)
print('VBench I2V Evaluation Results:')
print('=' * 60)
for k, v in eval_info_dict.items():
    print(f'  {{k}}: {{v:.4f}}' if isinstance(v, float) else f'  {{k}}: {{v}}')
print('=' * 60)
print(f'Results saved to: {{results_path}}')
'''
    script_path = f"{OUTPUT_BASE}/_run_vbench_i2v_eval.py"
    with open(script_path, "w") as f:
        f.write(eval_script)

    log_file = f"{OUTPUT_BASE}/vbench_i2v_eval.log"
    cmd = f"source {CONDA_PATH}/bin/activate fastvideo && cd /root/data/FAR-World && CUDA_VISIBLE_DEVICES=0 python {script_path}"

    print("[Evaluation] Running VBench I2V evaluation...")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT)

    # Wait for evaluation to complete
    proc.wait()

    # Print results
    results_path = f"{video_dir}/vbench_info/final_results.json"
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        print("\n" + "=" * 60)
        print("VBench I2V Evaluation Results:")
        print("=" * 60)
        for k, v in results.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print("=" * 60)
    else:
        print(f"[Evaluation] Results not found. Check log: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="VBench I2V evaluation for FastVideo")
    parser.add_argument("--machine", type=int, default=0, choices=[0, 1],
                        help="Machine ID: 0 for first half (samples 0-2794), 1 for second half (samples 2795-5589)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip VBench evaluation and upload (for multi-machine runs)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    fastvideo_i2v_output = f"{OUTPUT_BASE}/fastvideo_i2v"

    # Calculate sample range for this machine
    half = TOTAL_SAMPLES // 2
    if args.machine == 0:
        start_sample = 0
        end_sample = half
    else:
        start_sample = half
        end_sample = TOTAL_SAMPLES

    samples_for_machine = end_sample - start_sample

    print("=" * 60)
    print(f"VBench I2V Evaluation - FastVideo (Machine {args.machine})")
    print(f"Total prompts: {NUM_PROMPTS}")
    print(f"Samples per prompt: {NUM_SAMPLES_PER_PROMPT}")
    print(f"Total videos: {TOTAL_SAMPLES}")
    print(f"This machine: samples {start_sample}-{end_sample-1} ({samples_for_machine} videos)")
    print(f"Image directory: {IMAGE_DIR}")
    print("=" * 60)

    # Check if image directory exists
    if not os.path.exists(IMAGE_DIR):
        print(f"ERROR: Image directory not found: {IMAGE_DIR}")
        print("Please download the VBench I2V images first.")
        return

    # Phase 1: Generation with 8 workers (BF16, full GPU, parallel start)
    print(f"\n[Phase 1] Starting FastVideo I2V generation with 8 workers (BF16, full GPU)...")

    NUM_WORKERS = 8
    samples_per_worker = samples_for_machine // NUM_WORKERS
    procs = []
    for i in range(NUM_WORKERS):
        # Each worker handles samples_per_worker samples starting from start_sample
        worker_offset = start_sample + i * samples_per_worker
        proc = run_fastvideo_i2v_worker(i, i, worker_offset, samples_for_machine)
        procs.append(proc)

    # Monitor progress
    while True:
        all_done = all(p.poll() is not None for p in procs)
        video_count = count_videos(fastvideo_i2v_output)

        print(f"\r[Progress] FastVideo I2V (Machine {args.machine}): {video_count}/{samples_for_machine}", end="", flush=True)

        if all_done:
            print(f"\n[Phase 1] Generation complete! Total: {video_count}/{samples_for_machine}")
            break

        time.sleep(10)

    if not args.skip_eval:
        # Phase 2: VBench I2V Evaluation
        print("\n[Phase 2] Running VBench I2V evaluation...")
        run_vbench_i2v_evaluation(fastvideo_i2v_output)

        # Phase 3: Upload
        print("\n[Phase 3] Uploading results...")
        upload_results("fastvideo_i2v", fastvideo_i2v_output)
    else:
        print("\n[Skipped] VBench evaluation and upload (--skip-eval)")

    print("\n" + "=" * 60)
    print(f"VBench I2V FastVideo (Machine {args.machine}) complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
