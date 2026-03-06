# scrape_images.py
# Scrapes human images from the web to use as synthetic dataset
# Saves 256x256 JPEGs to data/train/1000/

import os
from pathlib import Path
from PIL import Image
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
from tqdm import tqdm

OUT_DIR = Path(__file__).parent.parent / "data" / "train" / "1000"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# diverse queries matching what SD would generate
QUERIES = [
    "person walking in park",
    "person sitting at desk indoors",
    "person standing on city street",
    "elderly person outdoors",
    "young person reading book",
    "person cooking in kitchen",
    "person jogging morning",
    "person standing in front of building",
    "child playing outdoors",
    "person sitting on bench",
]

TARGET     = 1000
PER_QUERY  = TARGET // len(QUERIES)  # 100 per query


def resize_and_save(src_dir, out_dir, start_idx):
    idx = start_idx
    for f in Path(src_dir).glob("*"):
        try:
            img = Image.open(f).convert("RGB").resize((256, 256), Image.LANCZOS)
            img.save(out_dir / f"{idx:05d}.jpg")
            idx += 1
        except Exception:
            pass  # skip corrupt images
    return idx


def main():
    existing = len(list(OUT_DIR.glob("*.jpg")))
    print(f"existing images: {existing} / {TARGET}")

    if existing >= TARGET:
        print("already have enough images, skipping.")
        return

    idx = existing
    tmp_dir = Path("/tmp/icrawler_tmp")

    for query in tqdm(QUERIES, desc="queries"):
        if idx >= TARGET:
            break

        tmp_dir.mkdir(parents=True, exist_ok=True)

        # try Bing first, fall back to Google
        try:
            crawler = BingImageCrawler(storage={"root_dir": str(tmp_dir)})
            crawler.crawl(keyword=query, max_num=PER_QUERY, file_idx_offset=0)
        except Exception:
            crawler = GoogleImageCrawler(storage={"root_dir": str(tmp_dir)})
            crawler.crawl(keyword=query, max_num=PER_QUERY, file_idx_offset=0)

        idx = resize_and_save(tmp_dir, OUT_DIR, idx)

        # clean up tmp
        for f in tmp_dir.glob("*"):
            f.unlink()

    total = len(list(OUT_DIR.glob("*.jpg")))
    print(f"\ndone. {total} total images in {OUT_DIR}")


if __name__ == "__main__":
    main()