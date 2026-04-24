# ML Pipeline: Re-training Pose Model

This folder contains a full retraining pipeline for custom human pose estimation with a focus on:

- stable full-body tracking,
- occlusions (hand covering camera/body parts),
- robust finger rays (`thumb`, `index`, `pinky`) for both hands,
- compatibility with export flows for browser inference.

## 1) What this pipeline trains

The pipeline trains a custom keypoint detector in YOLO Pose format (Ultralytics), then exports to ONNX.
You can later convert ONNX to runtime format used by your inference stack.

Target keypoints (23):

1. left_eye
2. right_eye
3. left_shoulder
4. right_shoulder
5. left_elbow
6. right_elbow
7. left_wrist
8. right_wrist
9. left_thumb
10. right_thumb
11. left_index
12. right_index
13. left_pinky
14. right_pinky
15. left_hip
16. right_hip
17. left_knee
18. right_knee
19. left_ankle
20. right_ankle
21. spine_center
22. abdomen_left
23. abdomen_right

## 2) Folder layout

`datasets/raw` - source datasets  
`datasets/processed` - converted YOLO-pose dataset  
`runs` - training artifacts  
`exports` - exported models  
`scripts` - all utility scripts  
`configs` - model/data config

## 3) Environment setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r ml-pipeline/requirements.txt
```

One-click full bootstrap (download + merge + prepare + augment):

```bash
ml-pipeline\run_bootstrap.bat
```

One-click automatic checks:

```bash
ml-pipeline\run_autocheck.bat
```

If you train on GPU, install a CUDA-enabled PyTorch build first, then run requirements install.

## 4) Dataset preparation

### 4.0 Auto-download public datasets

```bash
python ml-pipeline/scripts/download_public_datasets.py --try-crowdpose
```

This downloads COCO 2017 automatically and tries CrowdPose if the URL is available.

Merge multiple public datasets into one COCO-like source:

```bash
python ml-pipeline/scripts/merge_public_datasets.py ^
  --inputs ^
  "ml-pipeline/datasets/public/coco2017/train2017::ml-pipeline/datasets/public/coco2017/annotations/person_keypoints_train2017.json" ^
  "ml-pipeline/datasets/public/coco2017/val2017::ml-pipeline/datasets/public/coco2017/annotations/person_keypoints_val2017.json" ^
  --out-images ml-pipeline/datasets/raw/images ^
  --out-annotations ml-pipeline/datasets/raw/annotations.json
```

### 4.1 Collect raw data

Put raw data in:

- `ml-pipeline/datasets/raw/images`
- `ml-pipeline/datasets/raw/annotations.json`

Annotation format expected: COCO keypoints-like.

### 4.2 Convert and split

```bash
python ml-pipeline/scripts/prepare_dataset.py ^
  --raw-images ml-pipeline/datasets/raw/images ^
  --raw-annotations ml-pipeline/datasets/raw/annotations.json ^
  --out-dir ml-pipeline/datasets/processed ^
  --val-ratio 0.15 ^
  --seed 42
```

### 4.3 Add synthetic occlusions

```bash
python ml-pipeline/scripts/augment_occlusions.py ^
  --images-dir ml-pipeline/datasets/processed/images/train ^
  --labels-dir ml-pipeline/datasets/processed/labels/train ^
  --out-images-dir ml-pipeline/datasets/processed/images/train_aug ^
  --out-labels-dir ml-pipeline/datasets/processed/labels/train_aug ^
  --multiplier 1
```

Then merge `train` + `train_aug` (or train directly from `train_aug` if preferred).

## 4.4 Fully automatic pipeline run

```bash
python ml-pipeline/scripts/bootstrap_all.py --try-crowdpose
```

Quick experiment on a **random subset** of merged images (faster iteration on open data):

```bash
python ml-pipeline/scripts/bootstrap_all.py --max-images 8000
```

Useful flags:

- `--skip-install`
- `--skip-download`
- `--skip-merge`
- `--skip-prepare`
- `--skip-augment`
- `--augment-multiplier 2`

## 4.5 Automatic health checks

```bash
python ml-pipeline/scripts/autocheck.py --check-server --check-scripts --check-data
```

## 5) Training

```bash
python ml-pipeline/scripts/train.py ^
  --data ml-pipeline/configs/custom_pose.yaml ^
  --model yolo11n-pose.pt ^
  --imgsz 960 ^
  --epochs 200 ^
  --batch 16 ^
  --device 0 ^
  --project ml-pipeline/runs ^
  --name pose_opensource_v1
```

Training uses **open pretrained** YOLO pose weights (COCO) and stronger augmentations + `cos_lr` + tunable `--pose` loss gain; see `scripts/train.py --help`.

Recommended progression:

- quick baseline: `yolo11n-pose.pt`
- better quality: `yolo11s-pose.pt`
- high quality: `yolo11m-pose.pt`

## 6) Evaluate

```bash
python ml-pipeline/scripts/evaluate.py ^
  --weights ml-pipeline/runs/pose_occlusion_v1/weights/best.pt ^
  --data ml-pipeline/configs/custom_pose.yaml ^
  --imgsz 960 ^
  --device 0
```

## 7) Export

```bash
python ml-pipeline/scripts/export_model.py ^
  --weights ml-pipeline/runs/pose_opensource_v1/weights/best.pt ^
  --out-dir ml-pipeline/exports ^
  --imgsz 960
```

Exports ONNX and TorchScript and copies **`best.onnx`** into `--out-dir` for `POSE_MODEL_PATH`.

### Russian guide (открытые данные и шаги)

See **`ml-pipeline/docs/OBUCHENIE_OTKRITYE_DANNYE.md`**.

## 8) Integration checklist for current web app

1. Keep existing tracker rendering logic (already updated).
2. Use backend inference endpoint (`POST /api/inference/pose`) for model execution.
3. Return keypoints in the same names/order as this pipeline.
4. Reuse existing drawing code in `js/tracker.js` for lines and body topology.
5. Calibrate confidence threshold and smoothing parameters in production logs.

## 11) Server inference adapter

Implemented endpoint (Node server):

- `GET /api/inference/health`
- `POST /api/inference/pose` (multipart/form-data with `image`)

Environment variables:

- `PYTHON_BIN` (default: `python`)
- `POSE_MODEL_PATH` (default: `ml-pipeline/exports/best.onnx`)
- `POSE_IMGSZ` (default: `960`)
- `POSE_CONF` (default: `0.25`)
- `POSE_DEVICE` (default: `cpu`)
- `POSE_TIMEOUT_MS` (default: `6000`)

The adapter calls:

- `ml-pipeline/inference/pose_infer.py`

## 12) Live app switch to YOLO (server CPU)

The web app can run live keypoint inference through `POST /api/inference/pose` and automatically fallback to BlazePose if the endpoint/model is unavailable.

Recommended env for CPU mode:

```bash
set PYTHON_BIN=python
set POSE_MODEL_PATH=ml-pipeline/exports/best.onnx
set POSE_IMGSZ=960
set POSE_CONF=0.25
set POSE_DEVICE=cpu
set POSE_TIMEOUT_MS=6000
```

Health check:

```bash
GET /api/inference/health
```

Expected behavior in training page:

1. Try YOLO API first.
2. If API fails repeatedly or times out, switch to BlazePose.
3. Periodically re-check health and return to YOLO when available.

## 9) Training quality tips

- Add domain footage from your actual camera angle and room.
- Include hard negatives: partial body, fast motion blur, backlight.
- Ensure at least 20-30% samples include partial occlusion.
- Balance left/right hand visibility.
- Track per-keypoint AP (wrist/fingers/ankles usually fail first).

## 10) Notes

- This repository currently runs browser-side BlazePose.  
- Full replacement with custom retrained model requires inference integration (server or browser runtime conversion path).
- This pipeline gives you everything needed to train and version the model artifacts.
