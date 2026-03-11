import re
from pathlib import Path

import numpy as np
from icrawler.builtin import BingImageCrawler
from PIL import Image, ImageOps
import imagehash

# =============================
# CONFIG
# =============================
OUTPUT_DIR = Path("recyclable_waste_dataset")
TARGET_SIZE = (640, 640)      # YOLO-friendly
IMAGES_PER_CLASS = 50
PADDING_COLOR = (114, 114, 114)   # common neutral padding for detection datasets
MIN_DOWNLOAD_SIZE = (400, 400)

CLASSES = {
    "white_a4_paper": [
        "white A4 paper sheet real photo -cartoon -illustration -clipart -icon -drawing -toy -replica -render -crumpled -torn -dirty -damaged -used -waste -trash",
        "clean white office paper A4 real photograph -cartoon -illustration -clipart -icon -drawing -toy -replica -render -crumpled -torn -dirty -damaged -used -waste -trash",
        "blank white printer paper sheet real photo -cartoon -illustration -clipart -icon -drawing -toy -replica -render -crumpled -torn -dirty -damaged -used -waste -trash",
        "white paper sheet isolated real photo -cartoon -illustration -clipart -icon -drawing -toy -replica -render -crumpled -torn -dirty -damaged -used -waste -trash",
        "stack of white A4 paper real photo -cartoon -illustration -clipart -icon -drawing -toy -replica -render -crumpled -torn -dirty -damaged -used -waste -trash",
    ],
    "glass_bottle": [
        "glass bottle waste real photo -cartoon -illustration -clipart -icon -drawing -toy -replica -render",
        "empty glass bottle trash real photograph -cartoon -illustration -clipart -icon -drawing -toy -replica -render",
        "used glass bottle recycling real photo -cartoon -illustration -clipart -icon -drawing -toy -replica -render",
        "discarded glass bottle isolated real photo -cartoon -illustration -clipart -icon -drawing -toy -replica -render",
    ],
}

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

NEGATIVE_NAME_HINTS = {
    "cartoon", "illustration", "clipart", "icon", "drawing", "anime",
    "sketch", "painting", "logo", "symbol", "vector", "render", "3d",
    "toy", "replica", "figurine", "miniature", "cgi",
}

# =============================
# HELPERS
# =============================
def safe_name(s: str) -> str:
    s = s.strip().lower()
    return re.sub(r"[^a-z0-9_\-]+", "_", s).strip("_")

def list_images(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]

def letterbox_to_640(img: Image.Image, size=(640, 640)) -> Image.Image:
    """Resize with padding while keeping aspect ratio."""
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = ImageOps.contain(img, size, method=Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", size, PADDING_COLOR)
    x = (size[0] - img.size[0]) // 2
    y = (size[1] - img.size[1]) // 2
    canvas.paste(img, (x, y))
    return canvas

def process_resize_inplace(folder: Path, size=(640, 640)):
    """Resize all images to letterboxed JPG."""
    for p in list_images(folder):
        try:
            with Image.open(p) as im:
                out = letterbox_to_640(im, size=size)
                out_path = p.with_suffix(".jpg")
                out.save(out_path, format="JPEG", quality=90, optimize=True)

            if out_path != p:
                p.unlink(missing_ok=True)

        except Exception:
            p.unlink(missing_ok=True)

def is_probably_real_photo(im: Image.Image) -> bool:
    """
    Best-effort heuristic:
    Reject images that look overly flat / posterized / icon-like / cartoonish.
    This is NOT perfect, but helps reduce obvious non-photo images.
    """
    if im.mode != "RGB":
        im = im.convert("RGB")

    # Standardized preview
    preview = ImageOps.contain(im, (128, 128), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (128, 128), (255, 255, 255))
    x = (128 - preview.size[0]) // 2
    y = (128 - preview.size[1]) // 2
    canvas.paste(preview, (x, y))

    arr = np.asarray(canvas).astype(np.float32) / 255.0

    # Grayscale + simple edge strength
    gray = arr.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge_strength = float((gx.mean() + gy.mean()) / 2.0)

    # Quantized color diversity
    q16 = (arr * 15).astype(np.uint8)
    unique_colors = len(np.unique(q16.reshape(-1, 3), axis=0))

    # Saturation
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    sat = np.divide(
        maxc - minc,
        np.maximum(maxc, 1e-6),
        out=np.zeros_like(maxc),
        where=maxc > 1e-6
    )
    sat_mean = float(sat.mean())

    # Heuristics:
    # - icons/cartoons/renders often have low color diversity
    # - some also have low texture/edge complexity
    if unique_colors < 18:
        return False

    if unique_colors < 28 and edge_strength < 0.035:
        return False

    if sat_mean > 0.55 and unique_colors < 45:
        return False

    return True

def filter_non_photos(folder: Path):
    """
    Remove images that are likely non-photographic, too small, or suspicious.
    """
    removed = 0

    for p in list_images(folder):
        try:
            name = p.stem.lower()

            if any(term in name for term in NEGATIVE_NAME_HINTS):
                p.unlink(missing_ok=True)
                removed += 1
                continue

            with Image.open(p) as im:
                w, h = im.size

                if w < MIN_DOWNLOAD_SIZE[0] or h < MIN_DOWNLOAD_SIZE[1]:
                    p.unlink(missing_ok=True)
                    removed += 1
                    continue

                if not is_probably_real_photo(im):
                    p.unlink(missing_ok=True)
                    removed += 1
                    continue

        except Exception:
            p.unlink(missing_ok=True)
            removed += 1

    return removed

def dedupe_by_hash(folder: Path, hash_size=16):
    """
    Remove duplicates using perceptual hashing.
    Keeps the first occurrence of each hash.
    """
    seen = {}
    removed = 0

    for p in sorted(list_images(folder)):
        try:
            with Image.open(p) as im:
                h = imagehash.phash(im, hash_size=hash_size)
        except Exception:
            p.unlink(missing_ok=True)
            removed += 1
            continue

        if h in seen:
            p.unlink(missing_ok=True)
            removed += 1
        else:
            seen[h] = p

    return removed

def rename_sequential(folder: Path, class_name: str):
    imgs = sorted(list_images(folder))
    for i, p in enumerate(imgs, start=1):
        new_name = f"{class_name}_{i:04d}.jpg"
        new_path = folder / new_name

        if p.name != new_name:
            if new_path.exists():
                new_path.unlink()
            p.rename(new_path)

def count_images(folder: Path) -> int:
    return len(list_images(folder))

def download_more_images(folder: Path, class_name: str, queries: list[str], needed: int):
    """
    Download more images into folder (best effort).
    We overfetch a bit because filtering + dedupe will remove some files.
    """
    if needed <= 0:
        return

    current = count_images(folder)
    target = current + needed

    for q in queries:
        if count_images(folder) >= target:
            break

        remaining = target - count_images(folder)
        fetch_num = max(remaining * 2, 15)

        crawler = BingImageCrawler(
            storage={"root_dir": str(folder)},
            downloader_threads=6,
        )

        crawler.crawl(
            keyword=q,
            max_num=fetch_num,
            min_size=MIN_DOWNLOAD_SIZE,
        )

def build_class(class_name: str, queries: list[str]):
    folder = OUTPUT_DIR / safe_name(class_name)
    folder.mkdir(parents=True, exist_ok=True)

    attempts = 0
    max_attempts = 8   # increased because realism filter removes more images

    while count_images(folder) < IMAGES_PER_CLASS and attempts < max_attempts:
        need = IMAGES_PER_CLASS - count_images(folder)

        download_more_images(
            folder,
            class_name,
            queries,
            needed=max(need, 10)
        )

        process_resize_inplace(folder, size=TARGET_SIZE)
        removed_nonphotos = filter_non_photos(folder)
        removed_dupes = dedupe_by_hash(folder)

        attempts += 1
        print(
            f"[{class_name}] attempt {attempts}: "
            f"{count_images(folder)} kept | "
            f"removed non-photos={removed_nonphotos}, dupes={removed_dupes}"
        )

    # Final cleanup
    process_resize_inplace(folder, size=TARGET_SIZE)
    filter_non_photos(folder)
    dedupe_by_hash(folder)
    rename_sequential(folder, safe_name(class_name))

    final_n = count_images(folder)
    print(f"[{class_name}] final: {final_n}/{IMAGES_PER_CLASS} images -> {folder}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for class_name, queries in CLASSES.items():
        build_class(class_name, queries)

    print("\nDone.")
    print("Saved to:", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    main()