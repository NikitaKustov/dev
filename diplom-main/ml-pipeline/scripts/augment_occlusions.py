import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Create synthetic occlusion-augmented training images.")
    p.add_argument("--images-dir", type=str, required=True)
    p.add_argument("--labels-dir", type=str, required=True)
    p.add_argument("--out-images-dir", type=str, required=True)
    p.add_argument("--out-labels-dir", type=str, required=True)
    p.add_argument("--multiplier", type=int, default=1, help="How many augmented copies per image.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def apply_rectangle_occlusions(img, rng):
    h, w = img.shape[:2]
    holes = rng.randint(2, 6)
    out = img.copy()
    for _ in range(holes):
        hole_w = rng.randint(max(16, w // 30), max(32, w // 8))
        hole_h = rng.randint(max(16, h // 30), max(32, h // 8))
        x1 = rng.randint(0, max(0, w - hole_w))
        y1 = rng.randint(0, max(0, h - hole_h))
        color = rng.randint(0, 50)
        out[y1:y1 + hole_h, x1:x1 + hole_w] = color
    return out


def apply_grid_dropout(img, rng):
    h, w = img.shape[:2]
    out = img.copy()
    step = rng.randint(28, 64)
    drop = int(step * rng.uniform(0.3, 0.45))
    offset_x = rng.randint(0, step - 1)
    offset_y = rng.randint(0, step - 1)
    for y in range(offset_y, h, step):
        for x in range(offset_x, w, step):
            out[y:y + drop, x:x + drop] = rng.randint(0, 35)
    return out


def apply_shadow(img, rng):
    h, w = img.shape[:2]
    out = img.copy()
    shadow = np.zeros((h, w), dtype=np.uint8)
    x1, y1 = rng.randint(0, w // 2), rng.randint(0, h)
    x2, y2 = rng.randint(w // 2, w), rng.randint(0, h)
    x3, y3 = rng.randint(0, w), rng.randint(0, h // 2)
    pts = np.array([[x1, y1], [x2, y2], [x3, y3]], dtype=np.int32)
    cv2.fillPoly(shadow, [pts], 255)
    alpha = rng.uniform(0.25, 0.55)
    dark = (out * (1.0 - alpha)).astype(np.uint8)
    mask = shadow > 0
    out[mask] = dark[mask]
    return out


def apply_motion_blur(img, rng):
    k = rng.choice([3, 5, 7])
    kernel = np.zeros((k, k), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[k // 2, :] = 1.0 / k
    else:
        kernel[:, k // 2] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def apply_brightness_contrast(img, rng):
    alpha = rng.uniform(0.8, 1.2)
    beta = rng.randint(-25, 25)
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out


def main():
    args = parse_args()
    random.seed(args.seed)

    src_images = Path(args.images_dir)
    src_labels = Path(args.labels_dir)
    out_images = Path(args.out_images_dir)
    out_labels = Path(args.out_labels_dir)
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    image_paths = [p for p in src_images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    for img_path in tqdm(image_paths, desc="Augmenting"):
        label_path = src_labels / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # keep original copy in augmented folder too
        shutil.copy2(img_path, out_images / img_path.name)
        shutil.copy2(label_path, out_labels / label_path.name)

        for i in range(args.multiplier):
            aug_img = image.copy()
            choice = rng.choice(["rect", "grid", "shadow"])
            if choice == "rect":
                aug_img = apply_rectangle_occlusions(aug_img, rng)
            elif choice == "grid":
                aug_img = apply_grid_dropout(aug_img, rng)
            else:
                aug_img = apply_shadow(aug_img, rng)

            if rng.random() < 0.25:
                aug_img = apply_motion_blur(aug_img, rng)
            if rng.random() < 0.25:
                aug_img = apply_brightness_contrast(aug_img, rng)
            out_name = f"{img_path.stem}_occ_{i}{img_path.suffix}"
            cv2.imwrite(str(out_images / out_name), aug_img)
            shutil.copy2(label_path, out_labels / f"{img_path.stem}_occ_{i}.txt")

    print("Occlusion augmentation complete.")
    print(f"Images: {out_images}")
    print(f"Labels: {out_labels}")


if __name__ == "__main__":
    main()
