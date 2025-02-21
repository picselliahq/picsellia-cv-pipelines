import os
import random
import shutil

import albumentations as A
import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from picsellia import Experiment
from picsellia.sdk.asset import Asset
from picsellia.sdk.dataset import DatasetVersion
from picsellia.types.enums import LogType
from pycocotools.coco import COCO
from skimage.transform import resize

SIZE = 640


def get_classes_segmentation_dataset(segmentation_dataset: DatasetVersion) -> list[str]:
    labels = segmentation_dataset.list_labels()
    return [label.name for label in labels]


def get_classes_mask_dataset(mask_dataset: DatasetVersion) -> list[str]:
    labels = mask_dataset.list_labels()
    return [label.name for label in labels]


def split_train_test_val_filenames(
    image_files: list[str], seed: int
) -> tuple[list[str], list[str], list[str]]:
    random.Random(seed).shuffle(image_files)
    nbr_images = len(image_files)
    train_length = int(nbr_images * 0.8)
    test_length = (nbr_images - train_length) // 2
    train_image_filenames = image_files[:train_length]
    test_images_filenames = image_files[train_length : train_length + test_length]
    eval_images_filenames = image_files[train_length + test_length :]

    return train_image_filenames, test_images_filenames, eval_images_filenames


def makedirs_images_masks(
    x_train_dir: str,
    y_train_dir: str,
    x_test_dir: str,
    y_test_dir: str,
    x_eval_dir: str,
    y_eval_dir: str,
) -> None:
    os.makedirs(name=x_train_dir)
    os.makedirs(name=y_train_dir)

    os.makedirs(name=x_test_dir)
    os.makedirs(name=y_test_dir)

    os.makedirs(name=x_eval_dir)
    os.makedirs(name=y_eval_dir)


def move_images_and_masks_to_directories(
    image_path: str,
    mask_path: str,
    image_list: list[str],
    mask_list: list[str],
    dest_image_dir: str,
    dest_mask_dir: str,
    image_prefix: str,
    mask_prefix: str,
):
    for image_filename in tqdm.tqdm(image_list):
        try:
            mask_filename = _find_mask_by_image(
                image_filename=image_filename,
                mask_files=mask_list,
                mask_prefix=mask_prefix,
                image_prefix=image_prefix,
            )
        except ValueError:
            continue

        image_dest = os.path.join(dest_image_dir, image_filename)
        mask_dest = os.path.join(
            dest_mask_dir,
            _change_mask_filename_to_match_image(
                mask_prefix=mask_prefix.strip(),
                image_prefix=image_prefix.strip(),
                old_mask_filename=mask_filename,
            ),
        )

        img_file_path = os.path.join(image_path, image_filename)
        mask_file_path = os.path.join(mask_path, mask_filename)

        shutil.copy(img_file_path, image_dest)
        shutil.copy(mask_file_path, mask_dest)


def _find_mask_by_image(
    image_filename: str, image_prefix: str, mask_files: list[str], mask_prefix: str
) -> str:
    base_filename = image_filename.split(".")[0][len(image_prefix) :]
    for mask_file in mask_files:
        if base_filename == mask_file.split(".")[0][len(mask_prefix) :]:
            return mask_file
    raise ValueError(f"No mask found for image {image_filename}")


def _change_mask_filename_to_match_image(
    mask_prefix: str, image_prefix: str, old_mask_filename: str
) -> str:
    new_mask_filename = image_prefix + old_mask_filename[len(mask_prefix) :]

    return new_mask_filename


class Dataset:
    def __init__(
        self,
        images_dir,
        masks_dir,
        classes=None,
        augmentation=None,
        preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_filenames = [
            os.path.join(images_dir, image_id) for image_id in self.ids
        ]
        self.masks_filenames = [
            os.path.join(
                masks_dir,
                image_id.split(".")[0]
                + "."
                + get_mask_file_extension(os.listdir(masks_dir)[0]),
            )
            for image_id in self.ids
        ]
        self.class_values = [classes.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i) -> tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(self.images_filenames[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_filenames[i], 0)

        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype("float")

        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        return image, mask

    def get_image_filepath(self, i) -> str:
        return self.images_filenames[i]

    def __len__(self) -> int:
        return len(self.ids)


def get_mask_file_extension(mask_file_path: str) -> str:
    return mask_file_path.split(".")[1]


def extract_classes_from_mask(mask: np.ndarray, class_values: list) -> np.ndarray:
    masks = [(mask == v) for v in class_values]
    mask = np.stack(masks, axis=-1).astype("float")
    return mask


class Dataloader(keras.utils.Sequence):
    def __init__(self, dataset: Dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i) -> list:
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self) -> int:
        number_of_batches_per_epoch = len(self.indexes) // self.batch_size
        return number_of_batches_per_epoch

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


def get_training_augmentation() -> A.Compose:
    train_transform = [
        A.HorizontalFlip(p=0.3),
        A.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.2, border_mode=0
        ),
        A.Resize(SIZE, SIZE, p=1, always_apply=True),
        A.PadIfNeeded(
            min_height=SIZE, min_width=SIZE, always_apply=True, border_mode=0
        ),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.Sharpen(p=0.6),
                A.Blur(blur_limit=3, p=0.4),
                A.MotionBlur(blur_limit=3, p=0.4),
            ],
            p=0.4,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=0.7),
                A.HueSaturationValue(p=0.7),
            ],
            p=0.4,
        ),
        A.Lambda(mask=round_clip_0_1),
    ]
    return A.Compose(train_transform, is_check_shapes=False)


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


def get_validation_augmentation() -> A.Compose:
    test_transform = [
        A.PadIfNeeded(SIZE, SIZE),
        A.Resize(SIZE, SIZE, p=1, always_apply=True),
    ]
    return A.Compose(test_transform, is_check_shapes=False)


def get_preprocessing(preprocessing_fn) -> A.Compose:
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


def format_and_log_eval_metrics(experiment: Experiment, metrics: list, scores: list):
    eval_metrics = {"loss": float("{:.5f}".format(scores[0]))}
    for metric, value in zip(metrics, scores[1:]):
        eval_metrics[metric.__name__] = float("{:.5f}".format(value))

    experiment.log(name="eval-results", type=LogType.TABLE, data=eval_metrics)


def log_training_sample_to_picsellia(dataset: Dataset, experiment: Experiment):
    image_index_to_log = 0
    image_to_log, mask_to_log = dataset[image_index_to_log]
    image_filename = dataset.get_image_filepath(image_index_to_log)
    output_path = save_sample_file(
        output_path=os.path.join(experiment.png_dir, "output_image.jpg"),
        image=image_to_log,
        mask=mask_to_log[..., 0].squeeze(),
    )
    log_image_to_picsellia(
        file_path_to_log=output_path,
        experiment=experiment,
        log_name=f"sample-ground-truth-{str(os.path.basename(image_filename))}",
    )


def predict_and_log_mask(
    dataset: Dataset,
    experiment: Experiment,
    model,
):
    n = 2
    ids = np.random.choice(np.arange(len(dataset)), size=n)
    for i in ids:
        image, gt_mask = dataset[i]
        image_filename = dataset.get_image_filepath(i)
        image = np.expand_dims(image, axis=0)
        pr_mask = model.predict(image)

        output_path = save_sample_file(
            output_path=os.path.join(experiment.png_dir, "output_image_pred.jpg"),
            image=denormalize(image.squeeze()),
            gt_mask=gt_mask.squeeze(),
            pr_mask=pr_mask.squeeze(),
        )
        log_image_to_picsellia(
            file_path_to_log=output_path,
            experiment=experiment,
            log_name=f"sample-prediction-{str(os.path.basename(image_filename))}",
        )


def save_sample_file(output_path: str, **images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.savefig(output_path)
    return output_path


def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def log_image_to_picsellia(
    file_path_to_log: str, experiment: Experiment, log_name: str
):
    experiment.log(log_name, type=LogType.IMAGE, data=file_path_to_log)


def get_filename_from_fullpath(full_path: str) -> str:
    return full_path.split("/")[-1]


def shift_x_and_y_coordinates(polygon: np.ndarray) -> np.ndarray:
    shifted_contours = np.zeros_like(polygon)
    shifted_contours[:, 0] = polygon[:, 1]
    shifted_contours[:, 1] = polygon[:, 0]
    return shifted_contours


def format_polygons(polygons: list[np.ndarray]) -> list[list[list[int]]]:
    formatted_polygons = list(
        map(
            lambda polygon: list([int(coord[0]), int(coord[1])] for coord in polygon),
            polygons,
        )
    )
    return formatted_polygons


def move_files_for_polygon_creation(label_name: str, input_folder_path: str):
    new_folder_path = os.path.join(input_folder_path, label_name)
    os.makedirs(new_folder_path)
    files_to_move = [
        f
        for f in os.listdir(input_folder_path)
        if os.path.isfile(os.path.join(input_folder_path, f))
    ]
    for file_name in files_to_move:
        source_path = os.path.join(input_folder_path, file_name)
        destination_path = os.path.join(new_folder_path, file_name)

        shutil.copyfile(source_path, destination_path)


def find_asset_by_dataset_index(
    dataset: Dataset, dataset_version: DatasetVersion, i: int
) -> tuple[str, Asset]:
    image_filepath = dataset.get_image_filepath(i=i)
    asset = find_asset_from_path(image_path=image_filepath, dataset=dataset_version)
    return image_filepath, asset


def find_asset_from_path(image_path: str, dataset: DatasetVersion) -> Asset | None:
    asset_filename = get_filename_from_fullpath(image_path)
    try:
        asset = dataset.find_asset(filename=asset_filename)
        return asset
    except Exception as e:
        print(e)
        return None


def predict_mask_from_image(image: np.ndarray, model, asset: Asset) -> np.ndarray:
    image = np.expand_dims(image, axis=0)
    predicted_mask = model.predict(image)
    predicted_mask = resize(predicted_mask, (1, asset.height, asset.width, 1))

    return predicted_mask


def get_image_annotations(coco, image: dict) -> list:
    category_ids = coco.getCatIds()
    annotation_ids = coco.getAnnIds(
        imgIds=image["id"], catIds=category_ids, iscrowd=None
    )
    annotation_list = coco.loadAnns(annotation_ids)

    return annotation_list


def get_mask_from_annotations(coco: COCO, image_annotations: list) -> np.ndarray:
    mask = coco.annToMask(image_annotations[0])
    for i in range(len(image_annotations)):
        mask += coco.annToMask(image_annotations[i])
    return mask


def convert_mask_to_binary(mask: np.ndarray) -> np.ndarray:
    unique_values = np.unique(mask[mask != 0])
    converted_mask = np.where(np.isin(mask, unique_values), 255, 0)
    return converted_mask
