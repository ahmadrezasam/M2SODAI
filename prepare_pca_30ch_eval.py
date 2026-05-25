import argparse
import json
import os
import shutil

PROJECT_ROOT = "/home/ahmadreza/Downloads/Research/M2SODAI"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
HSI_NPY_ROOT = os.path.join(DATA_ROOT, "hsi_npy")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "baseline_official", "hsi_pca_30ch_eval")

SPLIT_CONFIG = {
    "val": {
        "coco": os.path.join(DATA_ROOT, "val_coco", "annotations_HSI.json"),
        "npy_dir": os.path.join(HSI_NPY_ROOT, "val_pca"),
    },
    "test": {
        "coco": os.path.join(DATA_ROOT, "test_coco", "annotations_HSI.json"),
        "npy_dir": os.path.join(HSI_NPY_ROOT, "test_pca"),
    },
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_coco(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _image_base(file_name: str) -> str:
    return os.path.splitext(os.path.basename(file_name))[0]


def _write_obb_labels(coco: dict, labels_dir: str, class_id: int = 0) -> list:
    images = {img["id"]: img for img in coco.get("images", [])}
    boxes_by_image = {img_id: [] for img_id in images}

    for ann in coco.get("annotations", []):
        if ann.get("category_id") != class_id:
            continue
        img = images.get(ann.get("image_id"))
        if not img:
            continue
        width = img.get("width")
        height = img.get("height")
        if not width or not height:
            continue
        x, y, bw, bh = ann["bbox"]
        x1 = x / width
        y1 = y / height
        x2 = (x + bw) / width
        y2 = y / height
        x3 = (x + bw) / width
        y3 = (y + bh) / height
        x4 = x / width
        y4 = (y + bh) / height
        boxes_by_image[ann["image_id"]].append([x1, y1, x2, y2, x3, y3, x4, y4])

    for img_id, img in images.items():
        base = _image_base(img["file_name"])
        label_path = os.path.join(labels_dir, f"{base}.txt")
        with open(label_path, "w") as f:
            for box in boxes_by_image.get(img_id, []):
                coords = " ".join(f"{v:.6f}" for v in box)
                f.write(f"0 {coords}\n")

    return [_image_base(img["file_name"]) for img in images.values()]


def _link_or_copy_npy(bases: list, npy_dir: str, images_dir: str, link: bool = True) -> None:
    missing = 0
    for base in bases:
        src = os.path.join(npy_dir, f"{base}.npy")
        dst = os.path.join(images_dir, f"{base}.npy")
        if os.path.exists(dst):
            continue
        if os.path.exists(src):
            if link:
                os.symlink(os.path.abspath(src), dst)
            else:
                shutil.copy2(src, dst)
        else:
            missing += 1
    if missing:
        print(f"Warning: {missing} PCA .npy files missing in {npy_dir}.")


def _write_yaml(yaml_path: str, root: str, split: str) -> None:
    with open(yaml_path, "w") as f:
        f.write(f"path: {root}\n")
        f.write(f"train: {split}/images\n")
        f.write(f"val: {split}/images\n")
        f.write("names:\n  0: ship\n")


def prepare_split(split: str, link: bool = True) -> None:
    cfg = SPLIT_CONFIG[split]
    coco_path = cfg["coco"]
    npy_dir = cfg["npy_dir"]

    if not os.path.exists(coco_path):
        raise FileNotFoundError(f"COCO annotation file not found: {coco_path}")
    if not os.path.isdir(npy_dir):
        raise FileNotFoundError(f"PCA .npy directory not found: {npy_dir}")

    images_dir = os.path.join(OUTPUT_ROOT, split, "images")
    labels_dir = os.path.join(OUTPUT_ROOT, split, "labels")
    _ensure_dir(images_dir)
    _ensure_dir(labels_dir)

    coco = _load_coco(coco_path)
    bases = _write_obb_labels(coco, labels_dir, class_id=0)
    _link_or_copy_npy(bases, npy_dir, images_dir, link=link)

    print(f"Prepared {split}: {len(bases)} images")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test", "all"], default="all")
    parser.add_argument("--copy", action="store_true", help="Copy .npy instead of symlinking")
    args = parser.parse_args()

    if args.split in ["val", "all"]:
        prepare_split("val", link=not args.copy)
    if args.split in ["test", "all"]:
        prepare_split("test", link=not args.copy)

    _write_yaml(os.path.join(PROJECT_ROOT, "baseline_official", "yolo26_pca30_val.yaml"), OUTPUT_ROOT, "val")
    _write_yaml(os.path.join(PROJECT_ROOT, "baseline_official", "yolo26_pca30_test.yaml"), OUTPUT_ROOT, "test")
    print("YAML files written under baseline_official/")


if __name__ == "__main__":
    main()
