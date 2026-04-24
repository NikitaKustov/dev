import argparse
import json
import shutil
from pathlib import Path

from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Merge multiple COCO-like datasets into one.")
    p.add_argument("--inputs", type=str, nargs="+", required=True, help="Pairs in format: images_dir::annotations_json")
    p.add_argument("--out-images", type=str, required=True)
    p.add_argument("--out-annotations", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out_images = Path(args.out_images)
    out_images.mkdir(parents=True, exist_ok=True)
    out_annotations = Path(args.out_annotations)
    out_annotations.parent.mkdir(parents=True, exist_ok=True)

    merged = {
        "info": {"description": "Merged public pose datasets"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    next_img_id = 1
    next_ann_id = 1
    category_written = False

    for input_pair in args.inputs:
        if "::" not in input_pair:
            raise ValueError(f"Invalid --inputs entry: {input_pair}")
        images_dir_s, ann_json_s = input_pair.split("::", 1)
        images_dir = Path(images_dir_s)
        ann_json = Path(ann_json_s)
        if not ann_json.exists():
            print(f"Skipping missing annotations: {ann_json}")
            continue

        with ann_json.open("r", encoding="utf-8") as f:
            src = json.load(f)

        if not category_written and src.get("categories"):
            merged["categories"] = src["categories"]
            category_written = True

        src_images = {img["id"]: img for img in src.get("images", [])}
        img_id_map = {}

        ds_prefix = ann_json.stem
        for img in tqdm(src.get("images", []), desc=f"Copy images from {ann_json.name}"):
            src_name = img["file_name"]
            src_path = images_dir / src_name
            if not src_path.exists():
                alt = images_dir / Path(src_name).name
                if alt.exists():
                    src_path = alt
                else:
                    continue

            new_file_name = f"{ds_prefix}_{next_img_id:09d}{src_path.suffix.lower()}"
            dst_path = out_images / new_file_name
            shutil.copy2(src_path, dst_path)

            new_img = dict(img)
            new_img["id"] = next_img_id
            new_img["file_name"] = new_file_name
            merged["images"].append(new_img)
            img_id_map[img["id"]] = next_img_id
            next_img_id += 1

        for ann in tqdm(src.get("annotations", []), desc=f"Copy anns from {ann_json.name}"):
            old_image_id = ann.get("image_id")
            if old_image_id not in img_id_map:
                continue
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = img_id_map[old_image_id]
            merged["annotations"].append(new_ann)
            next_ann_id += 1

    with out_annotations.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)

    print("Merged dataset saved.")
    print(f"images: {len(merged['images'])}")
    print(f"annotations: {len(merged['annotations'])}")
    print(f"out_images: {out_images.resolve()}")
    print(f"out_annotations: {out_annotations.resolve()}")


if __name__ == "__main__":
    main()
