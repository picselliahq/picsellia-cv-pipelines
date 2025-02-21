import numpy as np
import tqdm
import cv2
import os
from picsellia import DatasetVersion
from picsellia.types.enums import InferenceType


def convert_seperated_multiclass_masks_to_polygons(
    data_directory: str, dataset_version: DatasetVersion
):
    """

    Args:
        data_directory: (str)  directory containing the images. Example: data_path = "archive/input"
        dataset_version: (DatasetVersion) the dataset version containing the assets

    Returns: None

    """

    dataset_version.set_type(InferenceType.SEGMENTATION)
    mask_root_directory = "label_masks"
    input_dir = os.listdir(data_directory)
    labels = os.listdir(mask_root_directory)
    for fname in tqdm.tqdm(input_dir):
        asset = dataset_version.find_asset(filename=fname)
        polygons = []
        for label in labels:
            im = cv2.imread(os.path.join(mask_root_directory, label, fname))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(
                im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in contours:
                if len(c) > 3:
                    to_add = (
                        c[::1]
                        .reshape(
                            c[::1].shape[0],
                            c[::1].shape[2],
                        )
                        .tolist()
                    )
                    polygons.append(
                        (to_add, dataset_version.get_or_create_label(name=label))
                    )
        if len(polygons) > 0:
            try:
                annotation = asset.create_annotation(duration=0)
                annotation.create_multiple_polygons(polygons)
            except Exception as e:
                print(e)


def prepare_mask_directories_for_multilabel(class_to_pixel_mapping, mask_directory):
    """
    Create one directory per label, containing corresponding masks for that label
    Args:
        class_to_pixel_mapping (dict): mapping between labels and mask values. Example: {"car": 1, "plane": 63, "boat": 127}
        mask_directory (str): directory containing masks.

    Returns:

    """
    mask_root_directory = "label_masks"
    margin = compute_mask_margin(class_to_pixel_mapping)
    for key in class_to_pixel_mapping.keys():
        label_directory = os.path.join(mask_root_directory, key)
        os.makedirs(label_directory, exist_ok=True)  # create one directory per label

        for image_file in tqdm.tqdm(os.listdir(mask_directory)):
            image = cv2.imread(
                os.path.join(mask_directory, image_file), cv2.IMREAD_GRAYSCALE
            )

            pixel_value = int(class_to_pixel_mapping[key])
            masks = np.where(
                np.logical_and(
                    image <= (pixel_value + margin), image >= (pixel_value - margin)
                ),
                1,
                0,
            )
            new_mask_path = os.path.join(
                label_directory, image_file.split(".")[0] + ".jpg"
            )
            cv2.imwrite(new_mask_path, masks)  # save mask


def compute_mask_margin(class_to_pixel_mapping):
    pixel_values = list(class_to_pixel_mapping.values())
    pixel_values.sort()
    differences = []
    for i in range(len(pixel_values) - 1):
        differences.append(pixel_values[i + 1] - pixel_values[i])

    return int(min(differences) / 3)


def compute_class_to_pixel_dict(dataset_version: DatasetVersion):
    label_names = []
    for label in dataset_version.list_labels():
        label_names.append(label.name)
    len_labels = len(label_names)
    if len_labels > 24:
        raise Exception("You have more labels than this processing can support..")
    pixel_diff = int(255 / len_labels)
    class_to_pixel_dict = {label_names[0]: pixel_diff}

    for label in range(1, len_labels):
        class_to_pixel_dict[label_names[label]] = (
            class_to_pixel_dict[label_names[label - 1]] + pixel_diff
        )
    return class_to_pixel_dict
