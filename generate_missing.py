"""Generate missing/corrupted videos for VBench I2V evaluation."""
import os
import sys
sys.path.insert(0, '/root/data/video-gen-related/FastVideo')

from fastvideo import VideoGenerator, SamplingParam

MODEL_ID = "FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers"
IMAGE_DIR = "/root/data/FAR-World/vbench2_beta_i2v/data/crop/7-4"
OUTPUT_DIR = "/root/data/video-gen-related/outputs/vbench/fastvideo_i2v"
VBENCH_JSON = "/root/data/FAR-World/assets/data/meta/vbench/vbench2_i2v_aug_full_info.json"

NUM_FRAMES = 81
WIDTH = 1280
HEIGHT = 720
NEGATIVE_PROMPT = "Distorted, discontinuous, ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"

# 缺失的视频: (prompt_index, sample_idx)
MISSING = [
    (559, 1),  # two people in a canoe... seed=2797
    (2, 4),    # a close up of a blue and orange liquid... seed=15
]

def main():
    import json

    with open(VBENCH_JSON) as f:
        data = json.load(f)

    print("Loading model...")
    generator = VideoGenerator.from_pretrained(
        MODEL_ID,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_precision='bf16',
    )

    sampling_param = SamplingParam.from_pretrained(MODEL_ID)
    sampling_param.num_frames = NUM_FRAMES
    sampling_param.width = WIDTH
    sampling_param.height = HEIGHT
    sampling_param.negative_prompt = NEGATIVE_PROMPT

    for prompt_idx, sample_idx in MISSING:
        item = data[prompt_idx]
        aug_prompt = item.get('aug_prompt_en', item.get('prompt_en', ''))
        prompt_en = item.get('prompt_en', '')
        image_name = item.get('image_name', '')

        video_name = f"{prompt_en}-{sample_idx}.mp4"
        output_path = os.path.join(OUTPUT_DIR, video_name)

        if os.path.exists(output_path):
            print(f"Already exists: {video_name}")
            continue

        image_path = os.path.join(IMAGE_DIR, image_name)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        seed = prompt_idx * 5 + sample_idx + 1
        sampling_param.seed = seed

        print(f"\nGenerating: {video_name}")
        print(f"  Prompt idx: {prompt_idx}, Sample: {sample_idx}, Seed: {seed}")

        try:
            generator.generate_video(
                aug_prompt,
                image_path=image_path,
                output_path=output_path,
                save_video=True,
                sampling_param=sampling_param
            )
            print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone!")

if __name__ == "__main__":
    main()
