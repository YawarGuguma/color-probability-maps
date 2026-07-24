from pathlib import Path

import cv2
from ultralytics import YOLO

from yolo_sod import load_method, run_image, verify_model


root = Path(__file__).resolve().parents[1]
image = cv2.imread("example.jpg")
full_model = YOLO(root / "weights/wider/full_detector.pt")
patch_model = YOLO(root / "weights/wider/patch_detector.pt")
verify_model(full_model)
verify_model(patch_model)
method = load_method(root / "configs/wider")
result = run_image(full_model, patch_model, image, method)
print(result["combined"])
