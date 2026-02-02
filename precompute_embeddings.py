"""
Precompute all text embeddings for VBench I2V prompts.
This runs once before the main generation to avoid loading text encoder in each worker.

Usage:
    python precompute_embeddings.py
"""
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

VBENCH_I2V_JSON = "/root/data/FAR-World/assets/data/meta/vbench/vbench2_i2v_aug_full_info.json"
OUTPUT_DIR = "/root/data/video-gen-related/outputs/vbench/embeddings"

# UMT5-XXL model used by FastVideo Wan (d_model=4096)
# Load from FastVideo model's text_encoder directory
TEXT_ENCODER_PATH = "/root/data/tmp/hub/models--FastVideo--SFWan2.2-I2V-A14B-Preview-Diffusers/snapshots/0977572ccf137da5e577d62e3231ca840151af38/text_encoder"

NEGATIVE_PROMPT = '镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if already computed
    done_flag = os.path.join(OUTPUT_DIR, ".done")
    if os.path.exists(done_flag):
        print("Embeddings already precomputed. Delete .done file to recompute.")
        return

    print("Loading VBench I2V prompts...")
    with open(VBENCH_I2V_JSON, "r") as f:
        vbench_data = json.load(f)

    # Build all unique prompts (aug_prompt_en)
    prompts = []
    prompt_to_idx = {}
    for item in vbench_data:
        aug_prompt = item.get('aug_prompt_en', item.get('prompt_en', ''))
        if aug_prompt not in prompt_to_idx:
            prompt_to_idx[aug_prompt] = len(prompts)
            prompts.append(aug_prompt)

    print(f"Total unique prompts: {len(prompts)}")

    # Find tokenizer path
    tokenizer_path = TEXT_ENCODER_PATH.replace("text_encoder", "tokenizer")

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    print(f"Loading UMT5 encoder from: {TEXT_ENCODER_PATH}")
    model = UMT5EncoderModel.from_pretrained(TEXT_ENCODER_PATH, torch_dtype=torch.bfloat16, local_files_only=True)
    model = model.to("cuda")
    model.eval()

    print("Computing embeddings...")

    # Compute positive embeddings
    all_prompt_embeds = []
    all_attention_masks = []

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Encoding prompts"):
            inputs = tokenizer(
                prompt,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to("cuda")

            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )

            # Get last hidden state (shape: [1, 512, 4096])
            prompt_embeds = outputs.last_hidden_state.cpu()
            attention_mask = inputs.attention_mask.cpu()

            all_prompt_embeds.append(prompt_embeds)
            all_attention_masks.append(attention_mask)

    # Stack all embeddings
    all_prompt_embeds = torch.cat(all_prompt_embeds, dim=0)
    all_attention_masks = torch.cat(all_attention_masks, dim=0)

    print(f"Prompt embeddings shape: {all_prompt_embeds.shape}")  # Should be [N, 512, 4096]

    # Compute negative embedding (single)
    print("Computing negative prompt embedding...")
    with torch.no_grad():
        neg_inputs = tokenizer(
            NEGATIVE_PROMPT,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to("cuda")

        neg_outputs = model(
            input_ids=neg_inputs.input_ids,
            attention_mask=neg_inputs.attention_mask,
        )

        neg_prompt_embeds = neg_outputs.last_hidden_state.cpu()
        neg_attention_mask = neg_inputs.attention_mask.cpu()

    # Save everything
    output_path = os.path.join(OUTPUT_DIR, "embeddings.pt")
    print(f"Saving embeddings to {output_path}...")
    torch.save({
        'prompt_embeds': all_prompt_embeds,
        'attention_masks': all_attention_masks,
        'neg_prompt_embeds': neg_prompt_embeds,
        'neg_attention_mask': neg_attention_mask,
        'prompts': prompts,
        'prompt_to_idx': prompt_to_idx,
    }, output_path)

    # Save prompt mapping for reference
    with open(os.path.join(OUTPUT_DIR, "prompt_mapping.json"), "w") as f:
        json.dump(prompt_to_idx, f, ensure_ascii=False, indent=2)

    # Mark as done
    with open(done_flag, "w") as f:
        f.write("done")

    print(f"Embeddings saved to {OUTPUT_DIR}")
    print(f"  - prompt_embeds: {all_prompt_embeds.shape}")
    print(f"  - neg_prompt_embeds: {neg_prompt_embeds.shape}")
    print("You can now run the generation workers with skip_text_encoder=True")

    # Clean up
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
