#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import pyzipper


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--output",
        default=ROOT / "datasets/local_face/local_face_dataset.zip",
        type=Path,
    )
    parser.add_argument("--contact-email", required=True)
    return parser.parse_args()


def add_split(archive, source, source_names, output_name):
    directory = None
    for source_name in source_names:
        candidate = source / source_name
        if candidate.is_dir():
            directory = candidate
            break
    if directory is None:
        raise FileNotFoundError(source / source_names[0])
    for data_type in ("images", "labels"):
        data_directory = directory / data_type
        if not data_directory.is_dir():
            raise FileNotFoundError(data_directory)
        for path in sorted(data_directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(data_directory)
            archive_name = (
                Path("local_face_dataset")
                / data_type
                / output_name
                / relative
            )
            archive.write(path, str(archive_name))


def main():
    args = parse_args()
    password = os.environ.get("LOCAL_DATASET_PASSWORD", "")
    if len(password) < 12:
        raise RuntimeError(
            "Set LOCAL_DATASET_PASSWORD to a password containing at least 12 characters"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    readme = (
        "Controlled local face-detection dataset\n\n"
        f"Access contact: {args.contact_email}\n"
        "Use is limited to approved face-detection research.\n"
        "Redistribution, re-identification and password sharing are prohibited.\n"
        "The complete terms are provided in the accompanying GitHub repository.\n"
    )
    data_yaml = (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: face\n"
    )

    with pyzipper.AESZipFile(
        args.output,
        "w",
        compression=pyzipper.ZIP_DEFLATED,
        encryption=pyzipper.WZ_AES,
    ) as archive:
        archive.setpassword(password.encode("utf-8"))
        archive.setencryption(pyzipper.WZ_AES, nbits=256)
        add_split(archive, args.source, ("train",), "train")
        add_split(archive, args.source, ("val", "valid"), "val")
        add_split(archive, args.source, ("test",), "test")
        archive.writestr("local_face_dataset/local_face.yaml", data_yaml)
        archive.writestr("local_face_dataset/README.txt", readme)
        archive.write(
            ROOT / "datasets/local_face/split_manifest.csv",
            "local_face_dataset/split_manifest.csv",
        )
        archive.write(
            ROOT / "datasets/local_face/split_summary.json",
            "local_face_dataset/split_summary.json",
        )
    print(f"Saved encrypted dataset to {args.output.resolve()}")


if __name__ == "__main__":
    main()
