import os
import re
import json
import torch
from PIL import Image
from tqdm import tqdm
from typing import Optional
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
import argparse


def extract_background_description(text):
    patterns = [
        r"(?:\*\*)?Background Description:(?:\*\*)?\n\n(.*?)(?:\n\n---|\n\n\*\*Frames:)",
        r"Background Description:\n\n(.*?)(?=\n\nFrame_\d+:)",
        r"\*\*Background Description:\*\*\n(.*?)(?=\n\n\*\*Frames:\*\*)",
        r"(?:\*\*?)?Background Description:(?:\*\*?)?\n(.*?)(?=\n\n)",
        r"\*\*Background Description:\*\*\s*(.*?)\s*---",
        r"Background Description:\n(.*)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def load_pipeline(generator_type: str, cache_dir: str):
    if generator_type == "flux":
        return DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            cache_dir=cache_dir
        ).to("cuda")
    elif generator_type == "sdxl":
        return AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=cache_dir
        ).to("cuda")
    else:
        raise ValueError("Unsupported generator type. Choose 'flux' or 'sdxl'.")


def main(args):
    with open(args.plan_path, "r") as f:
        plan_bg = json.load(f)

    pipeline = load_pipeline(args.generator, args.cache_dir)
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    for k, v in plan_bg.items():
        skill_name = k.split(".")[0]

        if not os.path.exists(f"{output_path}/{skill_name}"):
            os.makedirs(f"{output_path}/{skill_name}", exist_ok=True)

        for i, plan in enumerate(tqdm(v)):
            index = i + 1
            bg = extract_background_description(plan)
            generator = torch.Generator("cuda").manual_seed(args.seed)

            if bg is not None:
                bg_image = pipeline(bg + ", realistic style", generator=generator).images[0]
                bg_image = bg_image.resize((1024, 1024))
            else:
                bg_image = Image.new("RGB", (1024, 1024), (255, 255, 255))
                print(f"Error processing background description for plan in {skill_name} index {i}")

            bg_image.save(f"{output_path}/{skill_name}/{index:04d}.jpg")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_path", type=str, required=True, help="Path to the background planning JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output background images directory")
    parser.add_argument("--generator", type=str, choices=["flux", "sdxl"], default="flux", help="Image generation pipeline to use")
    parser.add_argument("--seed", type=int, default=31, help="Seed for image generation")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache pretrained models")
    args = parser.parse_args()

    main(args)