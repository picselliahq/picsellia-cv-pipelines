import os

from picsellia_cv_engine.core.data.dataset.coco_dataset import CocoDataset
from picsellia_cv_engine.core.data.dataset.dataset_collection import DatasetCollection


def prepare_coco_directories(
    picsellia_datasets: DatasetCollection[CocoDataset],
) -> str:
    """Prepare COCO-format directory structure expected by YOLOX.

    YOLOX expects images in directories named train2017, val2017, test2017.
    This function creates symlinks from the dataset split image directories
    to the expected COCO directory names.

    Args:
        picsellia_datasets: Dataset collection with train/val/test splits.

    Returns:
        Base data directory path containing the symlinked split directories.
    """
    base_path = picsellia_datasets.dataset_path

    for split_name, coco_name in [
        ("train", "train2017"),
        ("val", "val2017"),
        ("test", "test2017"),
    ]:
        try:
            ds = picsellia_datasets[split_name]
        except KeyError:
            continue

        source_dir = ds.images_dir
        target_dir = os.path.join(base_path, coco_name)

        if not os.path.isdir(source_dir):
            raise RuntimeError(
                f"Source directory for '{split_name}' split does not exist: {source_dir}"
            )

        # YOLOX expects images at {data_dir}/{split}/images/{file_name}
        images_link = os.path.join(target_dir, "images")
        if not os.path.exists(images_link):
            os.makedirs(target_dir, exist_ok=True)
            os.symlink(source_dir, images_link)

    return base_path


def get_annotation_paths(
    picsellia_datasets: DatasetCollection[CocoDataset],
) -> dict[str, str]:
    """Get paths to COCO annotation JSON files for each split.

    Args:
        picsellia_datasets: Dataset collection with train/val/test splits.

    Returns:
        Dictionary mapping split names to their COCO annotation file paths.
    """
    paths = {}
    for split_name in ["train", "val", "test"]:
        try:
            ds = picsellia_datasets[split_name]
            paths[split_name] = ds.coco_file_path
        except KeyError:
            continue
    return paths
