import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = ROOT / "ml-pipeline"
SCRIPTS = ML_ROOT / "scripts"


def run(cmd, cwd=None):
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def parse_args():
    p = argparse.ArgumentParser(description="One-click bootstrap for full ML pipeline.")
    p.add_argument("--skip-install", action="store_true", help="Skip pip install -r requirements.txt")
    p.add_argument("--skip-download", action="store_true", help="Skip public dataset downloading")
    p.add_argument("--try-crowdpose", action="store_true", help="Try additional CrowdPose source")
    p.add_argument("--skip-merge", action="store_true", help="Skip merge public datasets")
    p.add_argument("--skip-prepare", action="store_true", help="Skip conversion to YOLO pose format")
    p.add_argument("--skip-augment", action="store_true", help="Skip occlusion augmentation")
    p.add_argument("--lite", action="store_true", help="Use lightweight public dataset bootstrap")
    p.add_argument(
        "--datasets-root",
        type=str,
        default="",
        help="Optional datasets root (e.g. E:\\ml-datasets). If provided, public/raw/processed will be stored there.",
    )
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment-multiplier", type=int, default=1)
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Передать в prepare_dataset: ограничить число кадров (ускоренное обучение на подвыборке COCO).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    py = sys.executable

    requirements = ML_ROOT / "requirements.txt"
    datasets_base = Path(args.datasets_root) if args.datasets_root else (ML_ROOT / "datasets")
    public_root = datasets_base / "public"
    raw_images = datasets_base / "raw" / "images"
    raw_annotations = datasets_base / "raw" / "annotations.json"
    processed_root = datasets_base / "processed"

    for d in [
        datasets_base / "raw",
        datasets_base / "processed",
        ML_ROOT / "exports",
        ML_ROOT / "runs",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    if not args.skip_install:
        run([py, "-m", "pip", "install", "-r", str(requirements)])

    if not args.skip_download:
        cmd = [py, str(SCRIPTS / "download_public_datasets.py"), "--out-dir", str(public_root)]
        if args.lite:
            cmd.append("--lite")
        if args.try_crowdpose:
            cmd.append("--try-crowdpose")
        run(cmd)

    if args.lite:
        print("\nLite mode detected: merge/prepare steps are skipped (dataset is already in YOLO layout).")
        print(f"Lite dataset location: {public_root / 'lite' / 'coco8-pose'}")
    elif not args.skip_merge:
        merge_inputs = [
            f"{public_root / 'coco2017' / 'train2017'}::{public_root / 'coco2017' / 'annotations' / 'person_keypoints_train2017.json'}",
            f"{public_root / 'coco2017' / 'val2017'}::{public_root / 'coco2017' / 'annotations' / 'person_keypoints_val2017.json'}",
        ]
        run(
            [
                py,
                str(SCRIPTS / "merge_public_datasets.py"),
                "--inputs",
                *merge_inputs,
                "--out-images",
                str(raw_images),
                "--out-annotations",
                str(raw_annotations),
            ]
        )

    if (not args.lite) and (not args.skip_prepare):
        prep_cmd = [
            py,
            str(SCRIPTS / "prepare_dataset.py"),
            "--raw-images",
            str(raw_images),
            "--raw-annotations",
            str(raw_annotations),
            "--out-dir",
            str(processed_root),
            "--val-ratio",
            str(args.val_ratio),
            "--seed",
            str(args.seed),
        ]
        if args.max_images is not None:
            prep_cmd.extend(["--max-images", str(args.max_images)])
        run(prep_cmd)

    if (not args.lite) and (not args.skip_augment):
        run(
            [
                py,
                str(SCRIPTS / "augment_occlusions.py"),
                "--images-dir",
                str(processed_root / "images" / "train"),
                "--labels-dir",
                str(processed_root / "labels" / "train"),
                "--out-images-dir",
                str(processed_root / "images" / "train_aug"),
                "--out-labels-dir",
                str(processed_root / "labels" / "train_aug"),
                "--multiplier",
                str(args.augment_multiplier),
                "--seed",
                str(args.seed),
            ]
        )

    print("\nBootstrap completed.")
    print("Next steps:")
    print(f"1) Train: {py} ml-pipeline/scripts/train.py --data ml-pipeline/configs/custom_pose.yaml")
    print(f"2) Eval : {py} ml-pipeline/scripts/evaluate.py --weights ml-pipeline/runs/<run>/weights/best.pt --data ml-pipeline/configs/custom_pose.yaml")
    print(f"3) Export: {py} ml-pipeline/scripts/export_model.py --weights ml-pipeline/runs/<run>/weights/best.pt --out-dir ml-pipeline/exports")


if __name__ == "__main__":
    main()
