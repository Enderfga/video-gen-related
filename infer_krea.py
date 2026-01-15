"""
Krea realtime-video T2V inference (4-step Self-Forcing)
Consistent settings: 81 frames, 832x480, seeds [0, 1]
"""
import json
import os
import sys
import time
from pathlib import Path

# Set MODEL_FOLDER before imports
os.environ["MODEL_FOLDER"] = "/raid/fga/LongLive/wan_models"
os.chdir("/raid/fga/related/realtime-video")
sys.path.insert(0, "/raid/fga/related/realtime-video")

import torch
from safetensors.torch import load_file as safe_load_file

# Patch WanTextEncoder to use .pth file instead of .safetensors
def patched_text_encoder_init(self):
    from wan.modules.tokenizers import HuggingfaceTokenizer
    from wan.modules.t5 import umt5_xxl
    from settings import MODEL_FOLDER

    torch.nn.Module.__init__(self)

    self.text_encoder = umt5_xxl(
        encoder_only=True,
        return_tokenizer=False,
        dtype=torch.float32,
        device=torch.device('cuda')
    ).eval().requires_grad_(False)

    # Use .pth file instead of .safetensors
    pth_path = os.path.join(MODEL_FOLDER, "Wan2.1-T2V-1.3B", "models_t5_umt5-xxl-enc-bf16.pth")
    state_dict = torch.load(pth_path, map_location='cuda', weights_only=True)
    self.text_encoder.load_state_dict(state_dict)

    self.tokenizer = HuggingfaceTokenizer(
        name=os.path.join(MODEL_FOLDER, "Wan2.1-T2V-1.3B", "google", "umt5-xxl/"),
        seq_len=512,
        clean='whitespace'
    )

# Apply patch
from utils import wan_wrapper
wan_wrapper.WanTextEncoder.__init__ = patched_text_encoder_init

from release_server import load_merge_config, load_all, GenerateParams, GenerationSession

torch.set_grad_enabled(False)

# ===== Shared Config =====
NUM_FRAMES = 81
WIDTH = 832
HEIGHT = 480
SEEDS = [0, 1]
NEGATIVE_PROMPT = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
PROMPT_FILE = "/raid/fga/related/prompt.json"
OUTPUT_PATH = "/raid/fga/related/outputs/krea_4step"
# =========================

# Krea specific: num_blocks to get ~81 frames (9 blocks * 9 frames per block)
NUM_BLOCKS = 9

os.makedirs(OUTPUT_PATH, exist_ok=True)


def save_video_direct(pixels: torch.Tensor, output_path: Path, fps: int = 16):
    """Save video using torchvision"""
    import torchvision.io as io

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_tensor = pixels[0].permute(0, 2, 3, 1).cpu().clamp(0, 1)
    video_tensor = (video_tensor * 255).to(torch.uint8)
    io.write_video(
        str(output_path), video_tensor, fps=fps, video_codec="h264", options={"crf": "18"}
    )
    print(f"Saved: {output_path}")


def main():
    config_path = "/raid/fga/related/realtime-video/configs/self_forcing_server_14b.yaml"

    print("Loading models...")
    config = load_merge_config(config_path)
    models = load_all(config)
    print("Models loaded!")

    with open(PROMPT_FILE, "r") as f:
        prompts_data = json.load(f)

    for seed in SEEDS:
        seed_dir = Path(OUTPUT_PATH) / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for idx, item in enumerate(prompts_data):
            prompt = item["prompt"]
            print(f"\n[Seed {seed}] [{idx+1}/{len(prompts_data)}] {prompt[:60]}...")

            params = GenerateParams(
                prompt=prompt,
                width=WIDTH,
                height=HEIGHT,
                num_blocks=NUM_BLOCKS,
                seed=seed,
                kv_cache_num_frames=3,
            )

            all_frames = []

            def frame_callback(pixels, frame_ids, event):
                event.synchronize()
                cpu_pixels = pixels.cpu().add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
                all_frames.append(cpu_pixels)

            session = GenerationSession(
                params=params,
                config=config,
                frame_callback=frame_callback,
                models=models,
            )

            t_start = time.time()
            for block_idx in range(params.num_blocks):
                session.generate_block(models)

            combined_frames = torch.cat(all_frames, dim=1)
            # Trim to exactly NUM_FRAMES if needed
            if combined_frames.shape[1] > NUM_FRAMES:
                combined_frames = combined_frames[:, :NUM_FRAMES]
            print(f"Generated {combined_frames.shape[1]} frames in {time.time() - t_start:.2f}s")

            video_path = seed_dir / f"video_{idx:03d}.mp4"
            save_video_direct(combined_frames, video_path, fps=16)

            session.dispose()
            all_frames.clear()

    print(f"\nAll videos saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
