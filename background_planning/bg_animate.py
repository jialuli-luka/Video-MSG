import os
import re
import json
import torch
from PIL import Image
from tqdm import tqdm
from typing import Optional
from diffusers import CogVideoXImageToVideoPipeline
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


def load_pipeline(cache_dir: str) -> CogVideoXImageToVideoPipeline:
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        "THUDM/CogVideoX-5b-I2V",
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe


def main(args):
    with open(args.plan_path, "r") as f:
        plan_bg = json.load(f)

    pipe = load_pipeline(args.cache_dir)

    input_background_path = args.input_background_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    for k, v in plan_bg.items():
        skill_name = k.split(".")[0]

        if not os.path.exists(f"{output_path}/{skill_name}"):
            os.makedirs(f"{output_path}/{skill_name}", exist_ok=True)

        for i, plan in enumerate(tqdm(v)):
            index = i + 1
            bg = extract_background_description(plan)
            image_path = os.path.join(input_background_path, skill_name, f"{index:04d}.jpg")

            if not os.path.exists(image_path):
                print(f"Missing image: {image_path}")
                continue

            image = Image.open(image_path)

            if bg is None:
                os.makedirs(f"{output_path}/{skill_name}/{index:04d}", exist_ok=True)
                for j in range(49):
                    image.save(f"{output_path}/{skill_name}/{index:04d}/{j}.jpg")
                continue

            generator = torch.Generator("cuda").manual_seed(args.seed)
            video = pipe(image, "Static Camera, " + bg, use_dynamic_cfg=True, generator=generator)

            os.makedirs(f"{output_path}/{skill_name}/{index:04d}", exist_ok=True)
            for j, img in enumerate(video.frames[0]):
                img.save(f"{output_path}/{skill_name}/{index:04d}/{j}.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_path", type=str, required=True, help="Path to the background planning JSON file")
    parser.add_argument("--input_background_path", type=str, required=True, help="Path to background images")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory for generated videos")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache pretrained models")
    parser.add_argument("--seed", type=int, default=31, help="Random seed")
    args = parser.parse_args()

    main(args)
    



