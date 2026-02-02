"""VBench I2V 8-GPU parallel evaluation."""
import os
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

VIDEO_DIR = "/root/data/video-gen-related/outputs/vbench/fastvideo_i2v"
OUTPUT_DIR = "/root/data/video-gen-related/outputs/vbench/eval_results"
VBENCH_JSON = "/root/data/FAR-World/assets/data/meta/vbench/vbench2_i2v_full_info.json"
CONDA_PATH = "/root/FGA/miniconda3"

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
    return (metric - range_[0]) / (range_[1] - range_[0])


def run_single_dimension(dim: str, gpu_id: int):
    """Run evaluation for a single dimension on specified GPU."""
    script = f'''
import os
import json
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'
os.environ['VBENCH_CACHE_DIR'] = '/root/data/FAR-World/experiments/pretrained_models/vbench'

from vbench2_beta_i2v import VBenchI2V

evaluator = VBenchI2V('cuda:0', '{VBENCH_JSON}', '{OUTPUT_DIR}')

print(f'[GPU {gpu_id}] Evaluating {dim}...')
evaluator.evaluate(
    videos_path='{VIDEO_DIR}',
    name='{dim}',
    dimension_list=['{dim}'],
    resolution='480p',
    local=True,
)
print(f'[GPU {gpu_id}] {dim} done!')
'''

    script_path = f"{OUTPUT_DIR}/_eval_{dim}.py"
    with open(script_path, 'w') as f:
        f.write(script)

    log_path = f"{OUTPUT_DIR}/{dim}.log"
    cmd = f"source {CONDA_PATH}/bin/activate fastvideo && python {script_path}"

    with open(log_path, 'w') as log:
        proc = subprocess.run(
            ["bash", "-c", cmd],
            stdout=log,
            stderr=subprocess.STDOUT
        )

    return dim, proc.returncode


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("VBench I2V 8-GPU Parallel Evaluation")
    print("=" * 60)
    print(f"Video dir: {VIDEO_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Dimensions: {len(DIMENSIONS)}")
    print()

    # Check video count
    video_count = len([f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')])
    print(f"Videos found: {video_count}")
    print()

    # Assign dimensions to GPUs (round-robin)
    tasks = [(dim, i % 8) for i, dim in enumerate(DIMENSIONS)]

    print("Task assignment:")
    for dim, gpu in tasks:
        print(f"  GPU {gpu}: {dim}")
    print()

    # Run in parallel
    print("Starting parallel evaluation...")
    results = {}

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_single_dimension, dim, gpu): dim for dim, gpu in tasks}

        for future in as_completed(futures):
            dim = futures[future]
            try:
                dim, returncode = future.result()
                status = "OK" if returncode == 0 else f"FAILED ({returncode})"
                print(f"  {dim}: {status}")
            except Exception as e:
                print(f"  {dim}: ERROR - {e}")

    # Collect results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)

    eval_info_dict = {}
    for dim in DIMENSIONS:
        result_file = os.path.join(OUTPUT_DIR, f"{dim}_eval_results.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
            # Extract score
            if isinstance(data, list) and len(data) > 0:
                scores = [item[1] for item in data if isinstance(item, list) and len(item) > 1]
                if scores:
                    raw_score = sum(scores) / len(scores)
                    norm_score = norm(raw_score, dim)
                    eval_info_dict[dim] = {'raw': raw_score, 'norm': norm_score}
                    print(f"  {dim:25s}: {norm_score:.4f} (raw: {raw_score:.4f})")
        else:
            print(f"  {dim:25s}: NOT FOUND")

    # Calculate composite scores
    if eval_info_dict:
        quality_score = (
            eval_info_dict.get('subject_consistency', {}).get('norm', 0) +
            eval_info_dict.get('background_consistency', {}).get('norm', 0) +
            eval_info_dict.get('motion_smoothness', {}).get('norm', 0) +
            eval_info_dict.get('dynamic_degree', {}).get('norm', 0) * 0.5 +
            eval_info_dict.get('aesthetic_quality', {}).get('norm', 0) +
            eval_info_dict.get('imaging_quality', {}).get('norm', 0)
        ) / 5.5

        i2v_score = (
            eval_info_dict.get('i2v_subject', {}).get('norm', 0) +
            eval_info_dict.get('i2v_background', {}).get('norm', 0) +
            eval_info_dict.get('camera_motion', {}).get('norm', 0) * 0.1
        ) / 2.1

        overall_score = 0.5 * i2v_score + 0.5 * quality_score

        print("-" * 60)
        print(f"  {'Quality Score':25s}: {quality_score:.4f}")
        print(f"  {'I2V Score':25s}: {i2v_score:.4f}")
        print(f"  {'Overall Score':25s}: {overall_score:.4f}")

        # Save results
        final_results = {
            **{k: v['norm'] for k, v in eval_info_dict.items()},
            'quality_score': quality_score,
            'i2v_score': i2v_score,
            'overall_score': overall_score,
        }

        results_file = os.path.join(OUTPUT_DIR, "final_results.json")
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
