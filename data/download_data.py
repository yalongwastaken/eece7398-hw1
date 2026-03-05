# download_data.py
# Downloads cat training categories and full validation set from HuggingFace
# Images are saved in ImageFolder-compatible structure:
#   data/train/<label_idx>/image.jpg
#   data/val/<label_idx>/image.jpg

import os
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login
from PIL import Image
from tqdm import tqdm

# cat label indices (281-293 in ImageNet)
CAT_LABELS = list(range(281, 294))

DATA_DIR = Path(__file__).parent
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"


def already_downloaded(out_dir, min_images=100):
    return len(list(Path(out_dir).rglob("*.jpg"))) >= min_images


def save_split(dataset, out_dir, label_filter=None):
    out_dir = Path(out_dir)
    for item in tqdm(dataset):
        label = item["label"]
        if label_filter and label not in label_filter:
            continue
        label_dir = out_dir / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        # count existing to avoid overwriting
        idx = len(list(label_dir.glob("*.jpg")))
        item["image"].convert("RGB").save(label_dir / f"{idx:05d}.jpg")


def main():
    login(token=os.environ["HF_TOKEN"])

    print("loading dataset (streaming)...")
    ds = load_dataset(
        "evanarlian/imagenet_1k_resized_256",
        split="train",
        streaming=True,
    )

    if already_downloaded(TRAIN_DIR):
        print(f"train already downloaded, skipping...")
    else:
        print(f"saving cat training images to {TRAIN_DIR}...")
        save_split(ds, TRAIN_DIR, label_filter=set(CAT_LABELS))

    print("loading validation set...")
    val_ds = load_dataset(
        "evanarlian/imagenet_1k_resized_256",
        split="val",
        streaming=True,
    )

    if already_downloaded(VAL_DIR, min_images=49000):
        print("val already downloaded, skipping...")
    else:
        print(f"saving full validation set to {VAL_DIR}...")
        save_split(val_ds, VAL_DIR)

    # summary
    train_count = sum(1 for _ in TRAIN_DIR.rglob("*.jpg"))
    val_count = sum(1 for _ in VAL_DIR.rglob("*.jpg"))
    print(f"done. {train_count} training images, {val_count} validation images")


if __name__ == "__main__":
    main()