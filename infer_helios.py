"""
Helios-Distilled T2V inference (pyramid [2,2,2] steps)
Settings: 132 frames (33x4), 640x384, seeds [0, 1], fps=24
"""
import json
import os

import torch
from diffusers import AutoModel, HeliosPyramidPipeline
from diffusers.utils import export_to_video

# ===== Shared Config =====
NUM_FRAMES = 132  # must be multiple of 33 (chunk size)
WIDTH = 640
HEIGHT = 384
SEEDS = [0, 1]
NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
PROMPT_FILE = "/raid/fga/video-gen-related/eval_caption_t2v.json"
OUTPUT_PATH = "/raid/fga/video-gen-related/outputs/helios_distilled"
# =========================

MODEL_PATH = "/raid/fga/video-gen-related/Helios-Distilled"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def main():
    torch.set_grad_enabled(False)

    print("Loading Helios-Distilled pipeline...")
    vae = AutoModel.from_pretrained(MODEL_PATH, subfolder="vae", torch_dtype=torch.float32)
    pipeline = HeliosPyramidPipeline.from_pretrained(
        MODEL_PATH,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    pipeline.to("cuda")
    print("Pipeline loaded!")

    with open(PROMPT_FILE, "r") as f:
        prompts_data = json.load(f)

    for seed in SEEDS:
        seed_dir = os.path.join(OUTPUT_PATH, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        for idx, item in enumerate(prompts_data):
            prompt = item["prompt"]
            print(f"\n[Seed {seed}] [{idx+1}/{len(prompts_data)}] {prompt[:60]}...")

            generator = torch.Generator(device="cuda").manual_seed(seed)

            output = pipeline(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_frames=NUM_FRAMES,
                height=HEIGHT,
                width=WIDTH,
                pyramid_num_inference_steps_list=[2, 2, 2],
                guidance_scale=1.0,
                is_amplify_first_chunk=True,
                generator=generator,
            )

            save_path = os.path.join(seed_dir, f"video_{idx:03d}.mp4")
            export_to_video(output.frames[0], save_path, fps=24)
            print(f"Saved: {save_path}")

    print(f"\nAll videos saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
