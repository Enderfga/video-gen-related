"""
NVlabs rCM T2V inference (4-step)
Consistent settings: 81 frames, 832x480, seeds [0, 1]
"""
import json
import math
import os
import sys

sys.path.insert(0, "/raid/fga/related/rcm")

import torch
from einops import rearrange
from tqdm import tqdm

from imaginaire.utils.io import save_image_or_video
from imaginaire.lazy_config import LazyCall as L, LazyDict, instantiate
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.utils.model_utils import init_weights_on_device, load_state_dict
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from rcm.networks.wan2pt1 import WanModel

torch._dynamo.config.suppress_errors = True

# ===== Shared Config =====
NUM_FRAMES = 81
WIDTH = 832
HEIGHT = 480
SEEDS = [0, 1]
NEGATIVE_PROMPT = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
PROMPT_FILE = "/raid/fga/related/prompt.json"
OUTPUT_PATH = "/raid/fga/related/outputs/rcm_4step"
# =========================

# Model paths
DIT_PATH = "/raid/fga/related/rCM_Wan2.1_T2V_14B_480p.pt"
VAE_PATH = "/raid/fga/related/CausVid/Wan2.1_VAE.pth"
TEXT_ENCODER_PATH = "/raid/fga/related/CausVid/models_t5_umt5-xxl-enc-bf16.pth"

# rCM settings
NUM_STEPS = 4
SIGMA_MAX = 80

WAN2PT1_14B_T2V: LazyDict = L(WanModel)(
    dim=5120,
    eps=1e-06,
    ffn_dim=13824,
    freq_dim=256,
    in_dim=16,
    model_type="t2v",
    num_heads=40,
    num_layers=40,
    out_dim=16,
    text_len=512,
)

tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}

os.makedirs(OUTPUT_PATH, exist_ok=True)


def main():
    # Load model
    log.info(f"Loading DiT from {DIT_PATH}")
    with init_weights_on_device():
        net = instantiate(WAN2PT1_14B_T2V).eval()

    state_dict = load_state_dict(DIT_PATH)
    prefix_to_load = "net."
    state_dict_dit_compatible = {}
    for k, v in state_dict.items():
        if k.startswith(prefix_to_load):
            state_dict_dit_compatible[k[len(prefix_to_load) :]] = v
        else:
            state_dict_dit_compatible[k] = v
    net.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
    del state_dict, state_dict_dit_compatible
    log.success(f"Successfully loaded DiT")

    net.to(**tensor_kwargs).cpu()
    torch.cuda.empty_cache()

    tokenizer = Wan2pt1VAEInterface(vae_pth=VAE_PATH)

    with open(PROMPT_FILE, "r") as f:
        prompts_data = json.load(f)

    for seed in SEEDS:
        seed_dir = os.path.join(OUTPUT_PATH, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        for idx, item in enumerate(prompts_data):
            prompt = item["prompt"]
            print(f"\n[Seed {seed}] [{idx+1}/{len(prompts_data)}] {prompt[:60]}...")

            # Get text embedding
            text_emb = get_umt5_embedding(
                checkpoint_path=TEXT_ENCODER_PATH, prompts=prompt
            ).to(dtype=torch.bfloat16).cuda()
            clear_umt5_memory()

            condition = {"crossattn_emb": text_emb.to(**tensor_kwargs)}

            state_shape = [
                tokenizer.latent_ch,
                tokenizer.get_latent_num_frames(NUM_FRAMES),
                HEIGHT // tokenizer.spatial_compression_factor,
                WIDTH // tokenizer.spatial_compression_factor,
            ]

            generator = torch.Generator(device=tensor_kwargs["device"])
            generator.manual_seed(seed)

            init_noise = torch.randn(
                1,
                *state_shape,
                dtype=torch.float32,
                device=tensor_kwargs["device"],
                generator=generator,
            )

            mid_t = [1.5, 1.4, 1.0][: NUM_STEPS - 1]
            t_steps = torch.tensor(
                [math.atan(SIGMA_MAX), *mid_t, 0],
                dtype=torch.float64,
                device=init_noise.device,
            )
            t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

            x = init_noise.to(torch.float64) * t_steps[0]
            ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
            total_steps = t_steps.shape[0] - 1

            net.cuda()
            for i, (t_cur, t_next) in enumerate(
                tqdm(
                    list(zip(t_steps[:-1], t_steps[1:])),
                    desc="Sampling",
                    total=total_steps,
                )
            ):
                with torch.no_grad():
                    v_pred = net(
                        x_B_C_T_H_W=x.to(**tensor_kwargs),
                        timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                        **condition,
                    ).to(torch.float64)
                    x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                        *x.shape,
                        dtype=torch.float32,
                        device=tensor_kwargs["device"],
                        generator=generator,
                    )
            samples = x.float()
            net.cpu()

            video = tokenizer.decode(samples)
            video = (1.0 + video.clamp(-1, 1)) / 2.0

            save_path = os.path.join(seed_dir, f"video_{idx:03d}.mp4")
            save_image_or_video(
                rearrange(video, "b c t h w -> c t h (b w)"), save_path, fps=16
            )
            print(f"Saved: {save_path}")

    print(f"\nAll videos saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
