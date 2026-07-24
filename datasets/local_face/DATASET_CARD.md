# Local face-detection dataset

## Summary

The dataset contains 436 frames of size 1920 x 1080 collected from two local
vehicle recordings. It contains 350 face annotations. Frames without faces
are retained to represent outdoor background conditions.

| Split | Images | Positive images | Faces | Negative images |
|---|---:|---:|---:|---:|
| Train | 194 | 65 | 178 | 129 |
| Validation | 64 | 36 | 70 | 28 |
| Test | 178 | 64 | 102 | 114 |

The training and validation data come from recording 101409. The test data
come from recording 101943. This prevents frames from the test video entering
training.

## Face sizes

Face size is determined from normalized face height.

- Small: height <= 3% of image height
- Medium: 3% < height <= 6%
- Large: height > 6%

| Split | Small | Medium | Large |
|---|---:|---:|---:|
| Train | 158 | 16 | 4 |
| Validation | 63 | 6 | 1 |
| Test | 93 | 9 | 0 |

## Annotations

Annotations use YOLO text format:

```text
class_id center_x center_y width height
```

All coordinates are normalized to the image dimensions. The only class is
`face`.

## Access

The images and labels are distributed in an AES-256 encrypted ZIP. Access is
granted by the dataset owner after a research request and acceptance of the
data-use terms.
