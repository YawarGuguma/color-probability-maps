# Running commands

Run commands from the repository root:

```bash
cd /path/to/cpm_comp_code
```

## 1. Installation

Install Ultralytics:

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
python -m pip install -e .
cd /path/to/cpm_comp_code
```

Install the remaining packages:

```bash
bash commands/01_install.sh
```

Check the installation and checkpoints:

```bash
python tests/test_package.py
```

## 2. Detection

Run the WIDER-trained method:

```bash
bash commands/02_detect.sh /path/to/image_or_directory wider
```

Run the local-dataset method:

```bash
bash commands/02_detect.sh /path/to/image_or_directory local
```

Direct command:

```bash
python scripts/detect.py /path/to/image_or_directory \
  --profile wider \
  --output results/detections
```

## 3. WIDER FACE evaluation

Required structure:

```text
WIDER_ROOT/
├── WIDER_train/images/
├── WIDER_val/images/
├── WIDER_test/images/
└── eval_tools/ground_truth/
```

Evaluate the included WIDER checkpoints:

```bash
export WIDER_ROOT=/path/to/WIDER_ROOT
bash commands/03_evaluate_wider.sh
```

Direct command:

```bash
python scripts/evaluate_wider.py \
  --dataset /path/to/WIDER_ROOT \
  --full-weights weights/wider/full_detector.pt \
  --patch-weights weights/wider/patch_detector.pt \
  --config configs/wider \
  --output results/wider \
  --device 0 \
  --half
```

Outputs:

```text
results/wider/
├── metrics.json
├── metrics.csv
├── patch_decisions.csv
├── precision_recall.pdf
├── precision_recall.png
├── ap50_comparison.pdf
├── ap50_comparison.png
├── predictions_full/
└── predictions_yolo_sod/
```

WIDER test annotations are not public. Generate WIDER test predictions with:

```bash
python scripts/detect.py \
  /path/to/WIDER_ROOT/WIDER_test/images \
  --profile wider \
  --output results/wider_test
```

## 4. WIDER FACE full-detector training

WIDER training labels must be in YOLO text format beside the image split.
Set the dataset path in `configs/widerface.yaml`.

Train for 100 epochs:

```bash
bash commands/06_train_full_wider.sh
```

Direct command:

```bash
python scripts/train_full_detector.py \
  --data configs/widerface.yaml \
  --model configs/yolo_sod_p1p4.yaml \
  --standard-weights yolov5nu.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 4 \
  --device 0 \
  --workers 8 \
  --project runs \
  --name wider_full
```

Continue from a P1-P4 checkpoint:

```bash
python scripts/train_full_detector.py \
  --data configs/widerface.yaml \
  --weights /path/to/p1p4_checkpoint.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 4 \
  --device 0 \
  --name wider_full_resume
```

## 5. WIDER Cr and patch-detector training

Fit the Cr distribution using training images only:

```bash
mkdir -p runs/wider_method_config
cp -a configs/wider/. runs/wider_method_config/

python scripts/fit_cr_distribution.py \
  --images datasets/widerface/WIDER_train/images \
  --labels datasets/widerface/WIDER_train/labels \
  --output runs/wider_method_config/cr_distribution.json
```

Create patch training data:

```bash
python scripts/prepare_patch_dataset.py \
  --images datasets/widerface/WIDER_train/images \
  --labels datasets/widerface/WIDER_train/labels \
  --full-weights runs/wider_full/weights/best.pt \
  --config runs/wider_method_config \
  --output datasets/wider_patches \
  --split train \
  --device 0 \
  --half
```

Create patch validation data:

```bash
python scripts/prepare_patch_dataset.py \
  --images datasets/widerface/WIDER_val/images \
  --labels datasets/widerface/WIDER_val/labels \
  --full-weights runs/wider_full/weights/best.pt \
  --config runs/wider_method_config \
  --output datasets/wider_patches \
  --split val \
  --device 0 \
  --half
```

Train the patch detector for 50 epochs:

```bash
python scripts/train_patch_detector.py \
  --weights runs/wider_full/weights/best.pt \
  --data datasets/wider_patches/patch_dataset.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --project runs \
  --name wider_patch
```

Calibrate the five fuzzy rules using two held-out parts of the WIDER training
set. The official WIDER validation split remains reserved for evaluation.

```bash
python scripts/calibrate_fuzzy_rules.py \
  --fit-images /path/to/wider_training_calibration_fit/images \
  --tune-images /path/to/wider_training_calibration_tune/images \
  --full-weights runs/wider_full/weights/best.pt \
  --distribution runs/wider_method_config/cr_distribution.json \
  --ranker runs/wider_method_config/region_ranker.json \
  --merge-rules runs/wider_method_config/merge_rules.json \
  --settings runs/wider_method_config/settings.json \
  --output runs/wider_method_config/fuzzy5.json \
  --dataset-name wider \
  --device 0 \
  --half
```

Evaluate the retrained method:

```bash
python scripts/evaluate_wider.py \
  --dataset /path/to/WIDER_ROOT \
  --full-weights runs/wider_full/weights/best.pt \
  --patch-weights runs/wider_patch/weights/best.pt \
  --config runs/wider_method_config \
  --output results/wider_retrained \
  --device 0 \
  --half
```

## 6. Controlled local dataset extraction

Request the archive password using the instructions in
`datasets/local_face/DATA_ACCESS_REQUEST.md`.

Extract with 7-Zip:

```bash
7z x datasets/local_face/local_face_dataset.zip \
  -odatasets/local_face
```

The extracted path must be:

```text
datasets/local_face/local_face_dataset/
```

## 7. Local test evaluation

Evaluate the included local checkpoints:

```bash
bash commands/04_evaluate_local.sh
```

Use another extracted dataset location:

```bash
export LOCAL_DATASET=/path/to/local_face_dataset
bash commands/04_evaluate_local.sh
```

Direct command:

```bash
python scripts/evaluate_local.py \
  --dataset datasets/local_face/local_face_dataset \
  --full-weights weights/local/full_detector.pt \
  --patch-weights weights/local/patch_detector.pt \
  --config configs/local \
  --output results/local \
  --device 0 \
  --half
```

Outputs:

```text
results/local/
├── metrics.json
├── metrics.csv
├── patch_decisions.csv
├── precision_recall.pdf
├── precision_recall.png
├── face_size_ap50.pdf
├── face_size_ap50.png
├── predictions_full/
└── predictions_yolo_sod/
```

## 8. Local full-detector transfer learning

Transfer from the WIDER P1-P4 full detector:

```bash
python scripts/train_full_detector.py \
  --data configs/local_face.yaml \
  --weights weights/wider/full_detector.pt \
  --epochs 60 \
  --imgsz 960 \
  --batch 4 \
  --device 0 \
  --workers 4 \
  --patience 20 \
  --project runs \
  --name local_full
```

## 9. Local Cr and patch-detector training

Create a working configuration:

```bash
mkdir -p runs/local_method_config
cp -a configs/local/. runs/local_method_config/
```

Learn the local face/background Cr distribution:

```bash
python scripts/fit_cr_distribution.py \
  --images datasets/local_face/local_face_dataset/images/train \
  --labels datasets/local_face/local_face_dataset/labels/train \
  --output runs/local_method_config/cr_distribution.json
```

Create training patches:

```bash
python scripts/prepare_patch_dataset.py \
  --images datasets/local_face/local_face_dataset/images/train \
  --labels datasets/local_face/local_face_dataset/labels/train \
  --full-weights runs/local_full/weights/best.pt \
  --config runs/local_method_config \
  --output datasets/local_patches \
  --split train \
  --device 0 \
  --half
```

Create validation patches:

```bash
python scripts/prepare_patch_dataset.py \
  --images datasets/local_face/local_face_dataset/images/val \
  --labels datasets/local_face/local_face_dataset/labels/val \
  --full-weights runs/local_full/weights/best.pt \
  --config runs/local_method_config \
  --output datasets/local_patches \
  --split val \
  --device 0 \
  --half
```

Train the local patch detector:

```bash
python scripts/train_patch_detector.py \
  --weights runs/local_full/weights/best.pt \
  --data datasets/local_patches/patch_dataset.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --project runs \
  --name local_patch
```

Calibrate the five fuzzy rules. Local training images form the fit set and
local validation images form the tuning set. Test images remain unused.

```bash
python scripts/calibrate_fuzzy_rules.py \
  --fit-images datasets/local_face/local_face_dataset/images/train \
  --tune-images datasets/local_face/local_face_dataset/images/val \
  --full-weights runs/local_full/weights/best.pt \
  --distribution runs/local_method_config/cr_distribution.json \
  --ranker runs/local_method_config/region_ranker.json \
  --merge-rules runs/local_method_config/merge_rules.json \
  --settings runs/local_method_config/settings.json \
  --output runs/local_method_config/fuzzy5.json \
  --dataset-name local \
  --device 0 \
  --half
```

Evaluate the retrained local method on the test split:

```bash
python scripts/evaluate_local.py \
  --dataset datasets/local_face/local_face_dataset \
  --full-weights runs/local_full/weights/best.pt \
  --patch-weights runs/local_patch/weights/best.pt \
  --config runs/local_method_config \
  --output results/local_retrained \
  --device 0 \
  --half
```
