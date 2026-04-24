import argparse

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate custom pose model.")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--split", type=str, default="val")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        split=args.split,
        task="pose",
    )
    print(metrics)


if __name__ == "__main__":
    main()
