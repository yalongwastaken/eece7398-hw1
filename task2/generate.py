# generate.py
# Generates synthetic "human" images using Stable Diffusion 1.5
# Output: 256x256 JPEGs saved to out/human/<idx>.jpg

import argparse
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm

OUT_DIR = Path(__file__).parent.parent / "data" / "train" / "1000"

# diverse prompts to avoid homogeneous dataset
PROMPTS = [
    "a photo of a person walking in a park, realistic, natural lighting",
    "a photo of a person sitting at a desk indoors, realistic",
    "a photo of a person standing on a city street, realistic",
    "a photo of an elderly person outdoors, realistic, natural lighting",
    "a photo of a young person reading a book, realistic",
    "a photo of a person cooking in a kitchen, realistic",
    "a photo of a person jogging in the morning, realistic",
    "a photo of a person standing in front of a building, realistic",
    "a photo of a child playing outdoors, realistic, natural lighting",
    "a photo of a person sitting on a bench, realistic, sunny day",
]


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"generating {args.n_images} images -> {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # count existing images to resume if interrupted
    existing = len(list(OUT_DIR.glob("*.jpg")))
    print(f"found {existing} existing images, resuming from there...")

    n_per_prompt = args.n_images // len(PROMPTS)
    idx = existing

    for prompt in tqdm(PROMPTS, desc="prompts"):
        for _ in range(n_per_prompt):
            if idx >= args.n_images + existing:
                break
            result = pipe(
                prompt,
                height=512, width=512,
                num_inference_steps=args.steps,
                guidance_scale=7.5,
            )
            img = result.images[0].resize((256, 256), Image.LANCZOS)
            img.save(OUT_DIR / f"{idx:05d}.jpg")
            idx += 1

    print(f"\ndone. {len(list(OUT_DIR.glob('*.jpg')))} total images in {OUT_DIR}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=1000,
                   help="total number of images to generate (500-5000)")
    p.add_argument("--steps",    type=int, default=20,
                   help="diffusion steps per image (20 is a good balance)")
    return p.parse_args()


if __name__ == "__main__":
    main()