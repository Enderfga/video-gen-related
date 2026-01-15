"""
LightX2V CausVid T2V inference (9-step)
Consistent settings: 81 frames, 832x480, seeds [0, 1]
"""
import json
import os
import sys

# Add lightx2v to path
sys.path.insert(0, "/raid/fga/related/lightx2v")

import torch
from lightx2v.pipeline import LightX2VPipeline

# ===== Shared Config =====
NUM_FRAMES = 81
WIDTH = 832
HEIGHT = 480
SEEDS = [0, 1]
NEGATIVE_PROMPT = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
PROMPT_FILE = "/raid/fga/related/prompt.json"
OUTPUT_PATH = "/raid/fga/related/outputs/lightx2v_9step"
# =========================

LIGHTX2V_PATH = "/raid/fga/related/lightx2v"
MODEL_PATH = "/raid/fga/related/CausVid"
CONFIG_JSON = f"{LIGHTX2V_PATH}/configs/causvid/wan_t2v_causvid.json"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def main():
    torch.set_grad_enabled(False)

    # Initialize pipeline once
    print("Initializing LightX2V CausVid pipeline...")
    pipeline = LightX2VPipeline(
        task="t2v",
        model_path=MODEL_PATH,
        model_cls="wan2.1_causvid",
    )
    pipeline.create_generator(config_json=CONFIG_JSON)
    print("Pipeline initialized!")

    with open(PROMPT_FILE, "r") as f:
        prompts_data = json.load(f)

    for seed in SEEDS:
        seed_dir = os.path.join(OUTPUT_PATH, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        for idx, item in enumerate(prompts_data):
            prompt = item["prompt"]
            print(f"\n[Seed {seed}] [{idx+1}/{len(prompts_data)}] {prompt[:60]}...")

            save_path = os.path.join(seed_dir, f"video_{idx:03d}.mp4")

            pipeline.generate(
                seed=seed,
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                save_result_path=save_path,
            )
            print(f"Saved: {save_path}")

    print(f"\nAll videos saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
