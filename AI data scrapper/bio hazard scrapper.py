import os
import re
from pathlib import Path

from icrawler.builtin import BingImageCrawler
from PIL import Image, ImageOps
import imagehash

# =============================
# CONFIG
# =============================
OUTPUT_DIR = Path("medical_waste_dataset")
TARGET_SIZE = (640, 640)      # YOLO-friendly
IMAGES_PER_CLASS = 50

CLASSES = {
    "gloves": [
        "used medical gloves biohazard waste",
        "contaminated disposable gloves medical waste bin",
        "biohazard bag gloves hospital waste",
    ],
    "needle": [
        "needle syringe medical sharps biohazard waste",
        "sharps container needle syringe close up",
        "used syringe needle disposal container",
    ],
    "gauze": [
        "blood soaked gauze medical waste",
        "used gauze bandage biohazard bag",
        "gauze dressing medical waste disposal",
    ],
    "masks": [
        "used surgical mask medical waste biohazard",
        "contaminated face mask disposal medical waste bin",
        "biohazard bag used masks hospital waste",
    ],
}

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# =============================
# HELPERS
# =============================
def safe_name(s: str) -> str:
    s = s.strip().lower()
    return re.sub(r"[^a-z0-9_\-]+", "_", s).strip("_")

def list_images(folder: Path):
    return [p for p in folder.iterdir() if p.suffix.lower() in ALLOWED_EXTS and p.is_file()]

def letterbox_to_640(img: Image.Image, size=(640, 640)) -> Image.Image:
    """Resize with padding (keeps aspect ratio)."""
    # Ensure 3-channel RGB
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    # Fit image inside size
    img = ImageOps.contain(img, size, method=Image.Resampling.LANCZOS)

    # Create padded canvas
    canvas = Image.new("RGB", size, (0, 0, 0))
    x = (size[0] - img.size[0]) // 2
    y = (size[1] - img.size[1]) // 2
    canvas.paste(img, (x, y))
    return canvas

def process_resize_inplace(folder: Path, size=(640, 640)):
    """Resize all images to size (letterbox) and save as .jpg."""
    for p in list_images(folder):
        try:
            with Image.open(p) as im:
                out = letterbox_to_640(im, size=size)
                # Convert everything to JPG to standardize
                out_path = p.with_suffix(".jpg")
                out.save(out_path, format="JPEG", quality=90, optimize=True)
            if out_path != p:
                p.unlink(missing_ok=True)
        except Exception:
            # If unreadable/corrupt, delete it
            p.unlink(missing_ok=True)

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
                h = imagehash.phash(im, hash_size=hash_size)  # robust for near-duplicates
        except Exception:
            p.unlink(missing_ok=True)
            removed += 1
            continue

        if h in seen:
            # duplicate
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
        # Avoid collisions
        if p.name != new_name:
            if new_path.exists():
                new_path.unlink()
            p.rename(new_path)

def count_images(folder: Path) -> int:
    return len(list_images(folder))

def download_more_images(folder: Path, class_name: str, queries: list[str], needed: int):
    """
    Download 'needed' more images into folder (best-effort).
    """
    if needed <= 0:
        return

    # Use a crawler per query, stop once we reach needed (approximately)
    current = count_images(folder)
    target = current + needed

    for q in queries:
        if count_images(folder) >= target:
            break

        remaining = target - count_images(folder)
        crawler = BingImageCrawler(
            storage={"root_dir": str(folder)},
            downloader_threads=6,
        )
        crawler.crawl(
            keyword=q,
            max_num=remaining,
            min_size=(300, 300),
        )

def build_class(class_name: str, queries: list[str]):
    folder = OUTPUT_DIR / safe_name(class_name)
    folder.mkdir(parents=True, exist_ok=True)

    # Loop: download -> resize -> dedupe, until we reach IMAGES_PER_CLASS or attempts exhausted
    attempts = 0
    max_attempts = 6  # usually enough to hit 50 even after dedupe

    while count_images(folder) < IMAGES_PER_CLASS and attempts < max_attempts:
        need = IMAGES_PER_CLASS - count_images(folder)
        download_more_images(folder, class_name, queries, needed=max(need, 10))  # overfetch a bit

        # Standardize (resize + convert to jpg)
        process_resize_inplace(folder, size=TARGET_SIZE)

        # Dedupe (phash)
        dedupe_by_hash(folder)

        attempts += 1

    # Final tidy
    process_resize_inplace(folder, size=TARGET_SIZE)
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