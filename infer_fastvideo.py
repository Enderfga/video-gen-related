"""
FastVideo CausalWan2.2 T2V inference (8-step)
Consistent settings: 81 frames, 832x480, seeds [0, 1]
"""
import json
import os
import sys

sys.path.insert(0, "/root/data/video-gen-related/FastVideo")

from fastvideo import VideoGenerator, SamplingParam
from fastvideo.configs.pipelines.wan import SelfForcingWan2_2_T2V480PConfig

# ===== Shared Config =====
NUM_FRAMES = 81
WIDTH = 832
HEIGHT = 480
SEEDS = [42, 1]
NEGATIVE_PROMPT = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
PROMPT_FILE = "/root/data/video-gen-related/prompt.json"
OUTPUT_PATH = "/root/data/video-gen-related/outputs/fastvideo_8step"
# =========================

os.makedirs(OUTPUT_PATH, exist_ok=True)


def main():
    model_id = "/root/data/video-gen-related/CausalWan2.2-I2V-A14B-Preview-Diffusers"
    pipeline_config = SelfForcingWan2_2_T2V480PConfig.from_pretrained(model_id)

    generator = VideoGenerator.from_pretrained(
        model_path=model_id,
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=True,
        dit_precision="fp32",
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        dmd_denoising_steps=[1000, 850, 700, 550, 350, 275, 200, 125],  # 8 steps
        pin_cpu_memory=True,
        pipeline_config=pipeline_config,
    )

    sampling_param = SamplingParam.from_pretrained(model_id)
    sampling_param.num_frames = NUM_FRAMES
    sampling_param.width = WIDTH
    sampling_param.height = HEIGHT
    sampling_param.negative_prompt = NEGATIVE_PROMPT

    with open(PROMPT_FILE, "r") as f:
        prompts_data = json.load(f)

    for seed in SEEDS:
        sampling_param.seed = seed
        seed_dir = os.path.join(OUTPUT_PATH, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        for idx, item in enumerate(prompts_data):
            prompt = item["prompt"]
            print(f"\n[Seed {seed}] [{idx+1}/{len(prompts_data)}] {prompt[:60]}...")

            _ = generator.generate_video(
                prompt,
                output_path=seed_dir,
                save_video=True,
                sampling_param=sampling_param,
            )

    print(f"\nAll videos saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
