import argparse

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(
        description="Train custom 23-kpt pose model (YOLO) on подготовленных открытых данных (COCO и др.)."
    )
    p.add_argument("--data", type=str, required=True)
    p.add_argument(
        "--model",
        type=str,
        default="yolo11n-pose.pt",
        help="Предобученные веса Ultralytics (COCO pose) — открытый чекпоинт.",
    )
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--project", type=str, default="ml-pipeline/runs")
    p.add_argument("--name", type=str, default="pose_opensource_v1")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--no-cos-lr",
        action="store_true",
        help="Отключить косинусный decay LR (по умолчанию включён — лучше сходимость на длинных эпохах).",
    )
    p.add_argument(
        "--pose",
        type=float,
        default=24.0,
        help="Вес компоненты loss по keypoints (Ultralytics pose loss gain).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        task="pose",
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        optimizer="AdamW",
        lr0=0.0015,
        lrf=0.01,
        cos_lr=not args.no_cos_lr,
        warmup_epochs=5.0,
        close_mosaic=25,
        degrees=12.0,
        translate=0.18,
        scale=0.45,
        shear=3.0,
        perspective=0.001,
        fliplr=0.5,
        hsv_h=0.02,
        hsv_s=0.55,
        hsv_v=0.4,
        mosaic=1.0,
        amp=True,
        patience=40,
        cache=False,
        pose=args.pose,
    )


if __name__ == "__main__":
    main()
