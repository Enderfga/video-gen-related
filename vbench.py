"""
VBench evaluation script for rCM with 8-GPU distributed inference.

Usage:
    PYTHONPATH=. torchrun --nnodes=1 --nproc_per_node=8 vbench.py \
        --dit_path assets/checkpoints/rCM_Wan2.1_T2V_1.3B_480p.pt \
        --save_dir output/vbench_results
"""

import argparse
import json
import math
import os

import torch
import torch.distributed as dist
from einops import rearrange, repeat
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

# Model configs from official inference script
WAN2PT1_1PT3B_T2V: LazyDict = L(WanModel)(
    dim=1536,
    eps=1e-06,
    ffn_dim=8960,
    freq_dim=256,
    in_dim=16,
    model_type="t2v",
    num_heads=12,
    num_layers=30,
    out_dim=16,
    text_len=512,
)

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

dit_configs = {"1.3B": WAN2PT1_1PT3B_T2V, "14B": WAN2PT1_14B_T2V}

tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VBench evaluation for rCM")
    parser.add_argument("--model_size", choices=["1.3B", "14B"], default="1.3B", help="Size of the model to use")
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4, help="1~4 for timestep-distilled inference")
    parser.add_argument("--sigma_max", type=float, default=80, help="Initial sigma for rCM")
    parser.add_argument("--dit_path", type=str, required=True, help="Path to the DiT model checkpoint")
    parser.add_argument("--vae_path", type=str, default="assets/checkpoints/Wan2.1_VAE.pth", help="Path to the Wan2.1 VAE")
    parser.add_argument("--text_encoder_path", type=str, default="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth", help="Path to the umT5 text encoder")
    parser.add_argument("--num_frames", type=int, default=77, help="Number of frames to generate")
    parser.add_argument("--resolution", default="480p", type=str, help="Resolution of the generated output")
    parser.add_argument("--aspect_ratio", default="16:9", type=str, help="Aspect ratio of the generated output")
    parser.add_argument("--save_dir", type=str, default="output/vbench_results", help="Directory to save generated videos")
    parser.add_argument("--vbench_json", type=str, default="assets/data/meta/vbench/VBench_full_info.json", help="Path to VBench prompt JSON")
    parser.add_argument("--num_samples_per_prompt", type=int, default=5, help="Number of samples per prompt")
    return parser.parse_args()


def load_vbench_prompts(json_path: str, num_samples_per_prompt: int) -> list:
    """Load VBench prompts and create samples with seeds."""
    with open(json_path, 'r') as f:
        meta_data = json.load(f)

    samples = []
    for prompt_item in meta_data:
        prompt = prompt_item.get('prompt_en', prompt_item.get('prompt', ''))
        for idx in range(num_samples_per_prompt):
            samples.append({
                'prompt': prompt,
                'video_name': f"{prompt}-{idx}.mp4",
                'seed': len(samples),  # Global unique seed
            })
    return samples


@torch.no_grad()
def generate_video(net, tokenizer, text_emb, args, seed: int, generator: torch.Generator):
    """Generate a single video using rCM official sampling logic."""
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]

    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(args.num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]

    generator.manual_seed(seed)

    init_noise = torch.randn(
        1,
        *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )

    condition = {"crossattn_emb": text_emb.to(**tensor_kwargs)}

    # Official rCM timesteps for better visual quality
    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]
    t_steps = torch.tensor(
        [math.atan(args.sigma_max), *mid_t, 0],
        dtype=torch.float64,
        device=init_noise.device,
    )

    # Convert TrigFlow timesteps to RectifiedFlow
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    # Sampling steps
    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)

    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        v_pred = net(
            x_B_C_T_H_W=x.to(**tensor_kwargs),
            timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
            **condition
        ).to(torch.float64)

        x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
            *x.shape,
            dtype=torch.float32,
            device=tensor_kwargs["device"],
            generator=generator,
        )

    samples = x.float()
    video = tokenizer.decode(samples)

    # Convert to [0, 1] range
    video = (1.0 + video.clamp(-1, 1)) / 2.0
    return video


def main():
    args = parse_arguments()

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        log.info(f"Running VBench evaluation with {world_size} GPUs")
        os.makedirs(args.save_dir, exist_ok=True)

    dist.barrier()

    # Load model
    with init_weights_on_device():
        net = instantiate(dit_configs[args.model_size]).eval()

    state_dict = load_state_dict(args.dit_path)
    prefix_to_load = "net."
    state_dict_dit_compatible = {}
    for k, v in state_dict.items():
        if k.startswith(prefix_to_load):
            state_dict_dit_compatible[k[len(prefix_to_load):]] = v
        else:
            state_dict_dit_compatible[k] = v
    net.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
    del state_dict, state_dict_dit_compatible

    if local_rank == 0:
        log.success(f"Successfully loaded DiT from {args.dit_path}")

    net.to(**tensor_kwargs)

    # Load tokenizer (VAE)
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)

    # Load VBench prompts
    samples = load_vbench_prompts(args.vbench_json, args.num_samples_per_prompt)

    if local_rank == 0:
        log.info(f"Total samples: {len(samples)}, samples per GPU: ~{len(samples) // world_size}")

    # Distribute samples across GPUs
    samples_per_rank = samples[local_rank::world_size]

    # Generator for reproducibility
    generator = torch.Generator(device=tensor_kwargs["device"])

    # Process samples
    for sample in tqdm(samples_per_rank, desc=f"Rank {local_rank}", disable=(local_rank != 0)):
        prompt = sample['prompt']
        seed = sample['seed']
        video_name = sample['video_name']

        save_path = os.path.join(args.save_dir, video_name)

        # Skip if already exists
        if os.path.exists(save_path):
            continue

        # Get text embedding
        text_emb = get_umt5_embedding(
            checkpoint_path=args.text_encoder_path,
            prompts=prompt
        ).to(dtype=torch.bfloat16).cuda()

        # Generate video
        video = generate_video(net, tokenizer, text_emb, args, seed, generator)

        # Save video
        save_image_or_video(
            rearrange(video, "b c t h w -> c t h (b w)"),
            save_path,
            fps=16
        )

        # Clear text encoder memory after each sample
        clear_umt5_memory()

    dist.barrier()

    if local_rank == 0:
        log.success(f"VBench evaluation complete. Videos saved to {args.save_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
