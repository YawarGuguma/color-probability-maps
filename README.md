# YOLO-SOD

YOLO-SOD combines a modified YOLOv5 nano detector with learned Cr probability
maps and five fuzzy rules. The detector has P1, P2, P3 and P4 prediction heads.
A full-image detector runs on every image. A second paraller detector runs only on
selected Cr patches.

All copy-paste commands are collected in
[`RUN_COMMANDS.md`](RUN_COMMANDS.md).

## Installation

The tested setup uses Python 3.11 and CUDA-enabled PyTorch.

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
python -m pip install -e .

cd /path/to/cpm_comp_code
python -m pip install -r requirements.txt
```

The recorded environment is also available in `environment.yml`.

## Model definition

The modified architecture is:

```text
configs/yolo_sod_p1p4.yaml
```

Ultralytics accepts this file directly. Copying it into `site-packages` is not
required.

```bash
yolo detect train \
  model=configs/yolo_sod_p1p4.yaml \
  data=configs/widerface.yaml \
  epochs=100 \
  imgsz=640 \
  batch=4
```

The Ultralytics configuration location is:

```text
<python-environment>/site-packages/ultralytics/cfg/models/v5/yolo_sod_p1p4.yaml
```

Files copied into `site-packages` are removed when Ultralytics is reinstalled.

## Detection

WIDER-trained weights:

```bash
python scripts/detect.py image.jpg --profile wider --output results/detect
```

Local-dataset weights:

```bash
python scripts/detect.py image.jpg --profile local --output results/detect_local
```

Yellow boxes show selected patches. Green boxes show face detections.

## WIDER FACE evaluation

Expected dataset layout:

```text
WIDER_ROOT/
├── WIDER_train/images/
├── WIDER_val/images/
├── WIDER_test/images/
└── eval_tools/ground_truth/
    ├── wider_face_val.mat
    ├── wider_easy_val.mat
    ├── wider_medium_val.mat
    └── wider_hard_val.mat
```

Run:

```bash
python scripts/evaluate_wider.py \
  --dataset /path/to/WIDER_ROOT \
  --output results/wider
```

The evaluator implements the WIDER Easy, Medium and Hard protocol at IoU 0.50.
It saves prediction text files, AP values, PR data, PDF graphs and PNG graphs.
Reference results are stored in `results/reference/wider_reference.csv`.

## Controlled local dataset

The local dataset contains identifiable participants. Written consent was
obtained on paper. The images are distributed only as an encrypted archive.
Access instructions and terms are in `datasets/local_face/`.

Dataset access requests are handled by the corresponding author. Signed consent
forms are retained privately by the dataset owner.

After approval, extract the archive so that the directory is:

```text
datasets/local_face/local_face_dataset/
├── images/train/
├── images/val/
├── images/test/
├── labels/train/
├── labels/val/
└── labels/test/
```

Evaluate:

```bash
python scripts/evaluate_local.py \
  --dataset datasets/local_face/local_face_dataset \
  --output results/local
```

The evaluator counts negative test images, reports overall and face-size
AP@0.50, and saves CSV, JSON, prediction files, PDF graphs and PNG graphs.
Reference results are stored in `results/reference/local_reference.csv`.

## Training

Train the full-image P1-P4 detector:

```bash
python scripts/train_full_detector.py \
  --data configs/widerface.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 4 \
  --name wider_full
```

Fit the Cr face/background distribution from training images only:

```bash
python scripts/fit_cr_distribution.py \
  --images /path/to/train/images \
  --labels /path/to/train/labels \
  --output configs/custom/cr_distribution.json
```

Prepare selected training and validation patches:

```bash
python scripts/prepare_patch_dataset.py \
  --images /path/to/train/images \
  --labels /path/to/train/labels \
  --full-weights runs/wider_full/weights/best.pt \
  --config configs/custom \
  --output datasets/patches \
  --split train

python scripts/prepare_patch_dataset.py \
  --images /path/to/val/images \
  --labels /path/to/val/labels \
  --full-weights runs/wider_full/weights/best.pt \
  --config configs/custom \
  --output datasets/patches \
  --split val
```

Train the patch detector:

```bash
python scripts/train_patch_detector.py \
  --weights runs/wider_full/weights/best.pt \
  --data datasets/patches/patch_dataset.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 8
```

Calibrate the five fuzzy rules on held-out training data:

```bash
python scripts/calibrate_fuzzy_rules.py \
  --fit-images /path/to/calibration_fit/images \
  --tune-images /path/to/calibration_tune/images \
  --full-weights runs/wider_full/weights/best.pt \
  --distribution configs/custom/cr_distribution.json \
  --ranker configs/custom/region_ranker.json \
  --merge-rules configs/custom/merge_rules.json \
  --settings configs/custom/settings.json \
  --output configs/custom/fuzzy5.json
```

The learned ranker and merge classifiers require labelled calibration data.
The fitted parameters used for the reported experiments are included under
`configs/wider/` and `configs/local/`.

## Checks

```bash
python tests/test_package.py
```

## Licences

The combined repository, model integration and checkpoints are under AGPL-3.0. Ultralytics also offers an enterprise licence. See `LICENSE`
and `THIRD_PARTY_NOTICES.md`.

The standalone `cr_probability.py` and `fuzzy_rules.py` components are
available under the MIT terms in `LICENSE_CR_MIT`. Distribution as part of
this Ultralytics-dependent application remains subject to AGPL-3.0.

The local dataset is not covered by either software licence. Its separate
controlled-access terms are in `datasets/local_face/DATASET_LICENSE.md`.
