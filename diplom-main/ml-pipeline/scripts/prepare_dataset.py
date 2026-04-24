import argparse
import json
import random
import shutil
from pathlib import Path

from tqdm import tqdm


# Order must match ml-pipeline/configs/custom_pose.yaml
KEYPOINT_NAMES = [
    "left_eye",
    "right_eye",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_thumb",
    "right_thumb",
    "left_index",
    "right_index",
    "left_pinky",
    "right_pinky",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "spine_center",
    "abdomen_left",
    "abdomen_right",
]


def parse_args():
    p = argparse.ArgumentParser(description="Convert COCO-like keypoints to YOLO pose format.")
    p.add_argument("--raw-images", type=str, required=True)
    p.add_argument("--raw-annotations", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Ограничить число изображений (после shuffle) для быстрых прогонов на открытых данных.",
    )
    return p.parse_args()


def ensure_dirs(base: Path):
    for sub in [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ]:
        (base / sub).mkdir(parents=True, exist_ok=True)


def coco_visibility_to_yolo(v: int) -> int:
    if v <= 0:
        return 0
    if v == 1:
        return 1
    return 2


def safe_point(xs, ys, vs):
    if not xs or not ys:
        return 0.0, 0.0, 0
    x = sum(xs) / len(xs)
    y = sum(ys) / len(ys)
    v = 2 if any(v >= 2 for v in vs) else (1 if any(v == 1 for v in vs) else 0)
    return x, y, v


def to_custom_keypoints(coco_points):
    # coco_points: list of [x, y, v], expected 17 keypoints layout
    # indices: 0 nose, 1 left_eye, 2 right_eye, 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles
    def kp(idx):
        if idx >= len(coco_points):
            return (0.0, 0.0, 0)
        return coco_points[idx]

    left_eye = kp(1)
    right_eye = kp(2)
    left_shoulder = kp(5)
    right_shoulder = kp(6)
    left_elbow = kp(7)
    right_elbow = kp(8)
    left_wrist = kp(9)
    right_wrist = kp(10)
    left_hip = kp(11)
    right_hip = kp(12)
    left_knee = kp(13)
    right_knee = kp(14)
    left_ankle = kp(15)
    right_ankle = kp(16)

    # Synthetic fingers when absent in source dataset.
    # They are initialized from wrists to let model start with coarse hand direction.
    left_thumb = left_wrist
    right_thumb = right_wrist
    left_index = left_wrist
    right_index = right_wrist
    left_pinky = left_wrist
    right_pinky = right_wrist

    sx, sy, sv = safe_point(
        [left_shoulder[0], right_shoulder[0]],
        [left_shoulder[1], right_shoulder[1]],
        [left_shoulder[2], right_shoulder[2]],
    )
    hx, hy, hv = safe_point(
        [left_hip[0], right_hip[0]],
        [left_hip[1], right_hip[1]],
        [left_hip[2], right_hip[2]],
    )
    spine_center = (sx * 0.45 + hx * 0.55, sy * 0.45 + hy * 0.55, 2 if sv and hv else max(sv, hv))
    abdomen_left = (left_hip[0], (left_hip[1] + left_shoulder[1]) / 2 if left_hip[2] and left_shoulder[2] else left_hip[1], max(left_hip[2], left_shoulder[2]))
    abdomen_right = (right_hip[0], (right_hip[1] + right_shoulder[1]) / 2 if right_hip[2] and right_shoulder[2] else right_hip[1], max(right_hip[2], right_shoulder[2]))

    return [
        left_eye,
        right_eye,
        left_shoulder,
        right_shoulder,
        left_elbow,
        right_elbow,
        left_wrist,
        right_wrist,
        left_thumb,
        right_thumb,
        left_index,
        right_index,
        left_pinky,
        right_pinky,
        left_hip,
        right_hip,
        left_knee,
        right_knee,
        left_ankle,
        right_ankle,
        spine_center,
        abdomen_left,
        abdomen_right,
    ]


def main():
    args = parse_args()
    random.seed(args.seed)

    raw_images = Path(args.raw_images)
    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)

    with open(args.raw_annotations, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    image_ids = list(images.keys())
    random.shuffle(image_ids)
    if args.max_images is not None and len(image_ids) > args.max_images:
        image_ids = image_ids[: args.max_images]
        print(f"Subset mode: using {len(image_ids)} images (--max-images).")
    n_val = int(len(image_ids) * args.val_ratio)
    val_ids = set(image_ids[:n_val])

    for image_id in tqdm(image_ids, desc="Converting"):
        img_info = images[image_id]
        file_name = img_info["file_name"]
        width = float(img_info["width"])
        height = float(img_info["height"])

        split = "val" if image_id in val_ids else "train"
        src_img = raw_images / file_name
        dst_img = out_dir / "images" / split / Path(file_name).name
        dst_lbl = out_dir / "labels" / split / (Path(file_name).stem + ".txt")

        if not src_img.exists():
            continue

        shutil.copy2(src_img, dst_img)

        lines = []
        for ann in anns_by_image.get(image_id, []):
            if ann.get("iscrowd", 0) == 1:
                continue

            bbox = ann.get("bbox", None)
            kps = ann.get("keypoints", None)
            if bbox is None or kps is None:
                continue

            x, y, w, h = bbox
            if w <= 1 or h <= 1:
                continue

            coco_points = []
            for i in range(0, len(kps), 3):
                coco_points.append((float(kps[i]), float(kps[i + 1]), int(kps[i + 2])))

            custom_kps = to_custom_keypoints(coco_points)

            xc = (x + w / 2) / width
            yc = (y + h / 2) / height
            wn = w / width
            hn = h / height

            row = ["0", f"{xc:.6f}", f"{yc:.6f}", f"{wn:.6f}", f"{hn:.6f}"]
            for px, py, pv in custom_kps:
                row.extend([
                    f"{max(0.0, min(1.0, px / width)):.6f}",
                    f"{max(0.0, min(1.0, py / height)):.6f}",
                    str(coco_visibility_to_yolo(int(pv))),
                ])
            lines.append(" ".join(row))

        dst_lbl.write_text("\n".join(lines), encoding="utf-8")

    print("Done.")
    print(f"Output: {out_dir}")
    print(f"Keypoints per object: {len(KEYPOINT_NAMES)}")


if __name__ == "__main__":
    main()
