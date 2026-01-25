"""
Master script to run all 4 video generation projects in parallel on 4 GPUs.
- GPU 0: FastVideo CausalWan2.2 (8-step)
- GPU 1: NVlabs rCM (4-step)
- GPU 2: Krea realtime-video (4-step)
- GPU 3: LightX2V CausVid (9-step)

All use consistent settings:
- num_frames = 81
- width = 832, height = 480
- seeds = [0, 1]

Usage:
    # Run all in parallel (use with conda activate fastvideo first)
    python run_all.py

    # Run single project
    python run_all.py --only krea
"""
import subprocess
import os
import sys
from pathlib import Path

CONDA_PATH = "/root/FGA/miniconda3"

SCRIPTS = {
    "fastvideo": {
        "script": "/root/data/video-gen-related/infer_fastvideo.py",
        "gpu": 0,
        "name": "FastVideo (8-step)",
        "env": "fastvideo",
    },
    # "rcm": {
    #     "script": "/root/data/video-gen-related/infer_rcm.py",
    #     "gpu": 1,
    #     "name": "rCM (4-step)",
    #     "env": "longlive",  # rcm needs longlive for flash_attn
    # },
    "krea": {
        "script": "/root/data/video-gen-related/infer_krea.py",
        "gpu": 2,
        "name": "Krea (4-step)",
        "env": "fastvideo",
    },
    "lightx2v": {
        "script": "/root/data/video-gen-related/infer_lightx2v.py",
        "gpu": 3,
        "name": "LightX2V (9-step)",
        "env": "lightx2v",
    },
}


def run_single(name: str):
    """Run a single project using its specific conda environment"""
    config = SCRIPTS[name]
    gpu = config["gpu"]
    script = config["script"]
    env = config["env"]

    print(f"[{config['name']}] Starting on GPU {gpu} with env '{env}'...")
    log_file = Path(f"/root/data/video-gen-related/outputs/{name}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Use bash to activate conda and run script
    cmd = f"source {CONDA_PATH}/bin/activate {env} && CUDA_VISIBLE_DEVICES={gpu} python {script}"

    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    return proc, config["name"], log_file


def run_all():
    """Run all 4 projects in parallel"""
    processes = []

    for name in SCRIPTS:
        proc, display_name, log_file = run_single(name)
        processes.append((proc, display_name, log_file))
        print(f"  -> {display_name} started, log: {log_file}")

    print("\nAll processes started. Waiting for completion...")

    for proc, display_name, log_file in processes:
        proc.wait()
        status = "✓ Done" if proc.returncode == 0 else f"✗ Failed (code {proc.returncode})"
        print(f"[{display_name}] {status}")

    print("\nAll tasks completed!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, choices=list(SCRIPTS.keys()),
                        help="Run only specific project")
    args = parser.parse_args()

    if args.only:
        proc, name, log = run_single(args.only)
        proc.wait()
        print(f"[{name}] Done, log: {log}")
    else:
        run_all()
