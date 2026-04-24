import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Export trained pose model.")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--imgsz", type=int, default=960)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz, opset=13, simplify=True))
    ts_path = Path(model.export(format="torchscript", imgsz=args.imgsz))

    best_onnx = out_dir / "best.onnx"
    shutil.copy2(onnx_path, best_onnx)
    shutil.copy2(Path(args.weights), out_dir / Path(args.weights).name)

    print("Export completed:")
    print(f"ONNX (source): {onnx_path}")
    print(f"ONNX (server default path): {best_onnx}")
    print(f"TorchScript: {ts_path}")
    print(f"Weights copy: {out_dir / Path(args.weights).name}")
    print("Для сервера: POSE_MODEL_PATH=" + str(best_onnx.resolve()))


if __name__ == "__main__":
    main()
