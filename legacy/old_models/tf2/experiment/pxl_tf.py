import io

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageDraw
from pxl_utils import format_segmentation


def process_image(image_path):
    im = Image.open(image_path)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = dict(im._getexif().items())
        if exif[orientation] == 3:
            im = im.transpose(Image.ROTATE_180)
        elif exif[orientation] == 6:
            im = im.transpose(Image.ROTATE_270)
        elif exif[orientation] == 8:
            im = im.transpose(Image.ROTATE_90)
    except (AttributeError, KeyError, IndexError):
        pass  # No exif

    encoded_jpg = io.BytesIO()
    try:
        im.save(encoded_jpg, format="JPEG")
    except OSError:
        im = im.convert("RGB")
        im.save(encoded_jpg, format="JPEG")
    return im, encoded_jpg.getvalue()


def handle_polygon_annotations(image, width, height):
    xmins, xmaxs, ymins, ymaxs, classes_text, classes, masks = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for a in image["annotations"]:
        if "segmentation" in a:
            poly = format_segmentation(a["segmentation"])
            poly = np.array(poly, dtype=np.float32)
            mask = np.zeros((height, width), dtype=np.uint8)
            mask = Image.fromarray(mask)
            ImageDraw.Draw(mask).polygon(poly, outline=1, fill=1)
            maskByteArr = io.BytesIO()
            mask.save(maskByteArr, format="JPEG")
            masks.append(maskByteArr.getvalue())

            if "bbox" not in a or len(a["bbox"]) == 0:
                (x, y, w, h) = cv2.boundingRect(poly)
            else:
                (x, y, w, h) = a["bbox"]

            xmins.append(np.clip(x / width, 0, 1))
            xmaxs.append(np.clip((x + w) / width, 0, 1))
            ymins.append(np.clip(y / height, 0, 1))
            ymaxs.append(np.clip((y + h) / height, 0, 1))
            classes_text.append(a["label"]["name"].encode("utf8"))
            classes.append(a["label"]["id"])

    return xmins, xmaxs, ymins, ymaxs, classes_text, classes, masks


def handle_rectangle_annotations(image, width, height):
    xmins, xmaxs, ymins, ymaxs, classes_text, classes = [], [], [], [], [], []

    for a in image["annotations"]:
        if "bbox" in a:
            (xmin, ymin, w, h) = a["bbox"]
            xmax = xmin + w
            ymax = ymin + h
            xmins.append(np.clip(xmin / width, 0, 1))
            xmaxs.append(np.clip(xmax / width, 0, 1))
            ymins.append(np.clip(ymin / height, 0, 1))
            ymaxs.append(np.clip(ymax / height, 0, 1))
            classes_text.append(a["label"]["name"].encode("utf8"))
            classes.append(a["label"]["id"])

    return xmins, xmaxs, ymins, ymaxs, classes_text, classes


def tf_vars_generator(annotations, label_map=None, annotation_type="rectangle"):
    if annotation_type not in ["polygon", "rectangle", "classification"]:
        raise ValueError("Please select a valid annotation_type")
    if label_map is None and annotation_type != "classification":
        raise ValueError("Provide a label_map when not working with classification")

    print(f"annotation type used for the variable generator: {annotation_type}")

    for image in annotations["images"]:
        im, encoded_jpg = process_image(image["path"])
        width, height = im.size
        filename = image["file_name"].encode("utf8")
        image_format = image["file_name"].split(".")[-1].encode("utf8")

        if annotation_type == "polygon":
            xmins, xmaxs, ymins, ymaxs, classes_text, classes, masks = (
                handle_polygon_annotations(image, width, height)
            )
            yield (
                width,
                height,
                xmins,
                xmaxs,
                ymins,
                ymaxs,
                filename,
                encoded_jpg,
                image_format,
                classes_text,
                classes,
                masks,
            )

        elif annotation_type == "rectangle":
            xmins, xmaxs, ymins, ymaxs, classes_text, classes = (
                handle_rectangle_annotations(image, width, height)
            )
            yield (
                width,
                height,
                xmins,
                xmaxs,
                ymins,
                ymaxs,
                filename,
                encoded_jpg,
                image_format,
                classes_text,
                classes,
            )
