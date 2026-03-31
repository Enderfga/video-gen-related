"""
Reorganize generated videos into t2v-distill-benchmark naming convention.

Convention: {model}/task_t2v/sample_step_{n}/seed_{s}/{idx:02d}_{prompt_prefix}.mp4
Prompt prefix: first ~40 chars of prompt, spaces replaced with underscores, truncated at word boundary.
"""
import json
import os
import re
import shutil
from pathlib import Path

PROMPT_FILE = "/raid/fga/video-gen-related/eval_caption_t2v.json"
OUTPUT_DIR = "/raid/fga/video-gen-related/outputs"
BENCHMARK_DIR = "/raid/fga/video-gen-related/t2v-distill-benchmark"

# Model configs: output_dir_name -> (benchmark_name, steps, seeds)
MODELS = {
    "fastvideo_8step": ("fastvideo", 8, [(42, 0), (1, 1)]),  # (src_seed, dst_seed)
    "rcm_4step": ("rcm_14b", 4, [0, 1]),
    "krea_4step": ("krea", 4, [0, 1]),
    "lightx2v_9step": ("lightx2v", 9, [0, 1]),
    "helios_distilled": ("helios", 6, [0, 1]),
}


def make_filename(idx, prompt, max_len=45):
    """Create filename like 00_a_vibrant_green_Mustang_GT_parked_in_a_p"""
    prefix = prompt[:max_len].replace(" ", "_").replace(",", ",").replace("'", "")
    # Clean special chars but keep basic punctuation
    prefix = re.sub(r'[^a-zA-Z0-9_,.\-]', '', prefix)
    return f"{idx:02d}_{prefix}.mp4"


def get_source_video(src_dir, idx, prompt):
    """Find the source video file - handles different naming conventions."""
    # Try video_XXX.mp4 (most models)
    v = os.path.join(src_dir, f"video_{idx:03d}.mp4")
    if os.path.exists(v):
        return v

    # Try prompt-based naming (FastVideo)
    for f in os.listdir(src_dir):
        if f.endswith(".mp4") and prompt[:40] in f:
            return os.path.join(src_dir, f)

    # Try matching by sorted order
    mp4s = sorted([f for f in os.listdir(src_dir) if f.endswith(".mp4")])
    if idx < len(mp4s):
        return os.path.join(src_dir, mp4s[idx])

    return None


def main():
    with open(PROMPT_FILE) as f:
        prompts = json.load(f)

    total_copied = 0

    for out_dir, (bench_name, steps, seeds) in MODELS.items():
        for seed_entry in seeds:
            if isinstance(seed_entry, tuple):
                src_seed, dst_seed = seed_entry
            else:
                src_seed, dst_seed = seed_entry, seed_entry
            src_dir = os.path.join(OUTPUT_DIR, out_dir, f"seed_{src_seed}")
            if not os.path.isdir(src_dir):
                continue

            dst_dir = os.path.join(BENCHMARK_DIR, bench_name, "task_t2v", f"sample_step_{steps}", f"seed_{dst_seed}")
            os.makedirs(dst_dir, exist_ok=True)

            for idx, item in enumerate(prompts):
                prompt = item["prompt"]
                src = get_source_video(src_dir, idx, prompt)
                if src is None:
                    continue

                dst_name = make_filename(idx, prompt)
                dst = os.path.join(dst_dir, dst_name)

                if os.path.exists(dst):
                    continue

                shutil.copy2(src, dst)
                total_copied += 1

            count = len([f for f in os.listdir(dst_dir) if f.endswith(".mp4")])
            print(f"  {bench_name}/seed_{dst_seed}: {count}/{len(prompts)} videos")

    print(f"\nTotal copied: {total_copied} videos")

    # Update metadata.json
    metadata = []
    for out_dir, (bench_name, steps, seeds) in MODELS.items():
        for seed_entry in seeds:
            dst_seed = seed_entry[1] if isinstance(seed_entry, tuple) else seed_entry
            dst_dir = os.path.join(BENCHMARK_DIR, bench_name, "task_t2v", f"sample_step_{steps}", f"seed_{dst_seed}")
            if not os.path.isdir(dst_dir):
                continue
            for idx, item in enumerate(prompts):
                dst_name = make_filename(idx, item["prompt"])
                dst = os.path.join(dst_dir, dst_name)
                if os.path.exists(dst):
                    metadata.append({
                        "file_name": f"{bench_name}/task_t2v/sample_step_{steps}/seed_{dst_seed}/{dst_name}",
                        "model": bench_name,
                        "steps": steps,
                        "seed": dst_seed,
                        "prompt_idx": idx,
                        "prompt": item["prompt"],
                    })

    meta_path = os.path.join(BENCHMARK_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata: {len(metadata)} entries written to {meta_path}")


if __name__ == "__main__":
    main()
