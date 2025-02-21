import io

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageDraw

from pxl_utils import format_segmentation


def tf_vars_generator(annotations, label_map=None, annotation_type="rectangle"):
    """THIS FUNCTION IS MAINTAINED FOR TENSORFLOW 1.X
    Generator for variable needed to instantiate a tf example needed for training.

    Args :
        label_map (tf format)
        ensemble (str) : Chose between train & test
        annotation_type: "polygon", "rectangle" or "classification"

    Yields :
        (width, height, xmins, xmaxs, ymins, ymaxs, filename,
               encoded_jpg, image_format, classes_text, classes, masks)

    Raises:
        ResourceNotFoundError: If you don't have performed your trained test split yet
                               If images can't be opened

    """
    if annotation_type not in ["polygon", "rectangle", "classification"]:
        raise ValueError("Please select a valid annotation_type")

    if label_map is None and annotation_type != "classification":
        raise ValueError(
            "Please provide a label_map dict loaded from a protobuf file when working with object detection"
        )

    if annotation_type == "classification":
        label_map = {v: int(k) for k, v in label_map.items()}

    # if annotation_type == "rectangle":
    #     for ann in annotations["annotations"]:
    #         for an in ann["annotations"]:
    #             if "segmentation" in an.keys() and len(an["segmentation"]) > 0:
    #                 annotation_type = "rectangle from polygon"
    #                 break

    print(f"annotation type used for the variable generator: {annotation_type}")

    for image in annotations["images"]:
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        masks = []

        im = Image.open(image["path"])
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
            # cases: image don't have getexif
            pass

        encoded_jpg = io.BytesIO()
        try:
            im.save(encoded_jpg, format="JPEG")
        except OSError:
            im = im.convert("RGB")
            im.save(encoded_jpg, format="JPEG")
        encoded_jpg = encoded_jpg.getvalue()

        width, height = im.size
        filename = image["file_name"].encode("utf8")
        image_format = image["file_name"].split(".")[-1]
        image_format = bytes(image_format.encode("utf8"))

        if annotation_type == "polygon":
            for a in image["annotations"]:
                if "segmentation" in a.keys():
                    poly = format_segmentation(a["segmentation"])
                    poly = np.array(poly, dtype=np.float32)
                    mask = np.zeros((height, width), dtype=np.uint8)
                    mask = Image.fromarray(mask)
                    ImageDraw.Draw(mask).polygon(poly, outline=1, fill=1)
                    maskByteArr = io.BytesIO()
                    mask.save(maskByteArr, format="JPEG")
                    maskByteArr = maskByteArr.getvalue()
                    masks.append(maskByteArr)

                    if "bbox" not in a.keys() or len(a["bbox"]) == 0:
                        (x, y, w, h) = cv2.boundingRect(poly)
                    else:
                        (x, y, w, h) = a["bbox"]
                    x1_norm = np.clip(x / width, 0, 1)
                    x2_norm = np.clip((x + w) / width, 0, 1)
                    y1_norm = np.clip(y / height, 0, 1)
                    y2_norm = np.clip((y + h) / height, 0, 1)

                    xmins.append(x1_norm)
                    xmaxs.append(x2_norm)
                    ymins.append(y1_norm)
                    ymaxs.append(y2_norm)
                    classes_text.append(a["label"]["name"].encode("utf8"))
                    # label_id = label_map[a["label"]]
                    label_id = a["label"]["id"]
                    classes.append(label_id)

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
            # for image_annoted in annotations["images"]:
            for a in image["annotations"]:
                try:
                    if "bbox" in a.keys():
                        (xmin, ymin, w, h) = a["bbox"]
                        xmax = xmin + w
                        ymax = ymin + h
                        ymins.append(np.clip(ymin / height, 0, 1))
                        ymaxs.append(np.clip(ymax / height, 0, 1))
                        xmins.append(np.clip(xmin / width, 0, 1))
                        xmaxs.append(np.clip(xmax / width, 0, 1))
                        classes_text.append(a["label"]["name"].encode("utf8"))
                        label_id = a["label"]["id"]
                        classes.append(label_id)
                except Exception as e:
                    print(f"An error occured with the image {filename}, {str(e)}")
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

        # if annotation_type == "classification":
        #     for image_annoted in dict_annotations["annotations"]:
        #         if internal_picture_id == image_annoted["internal_picture_id"]:
        #             for a in image_annoted["annotations"]:
        #                 classes_text.append(a["label"].encode("utf8"))
        #                 label_id = label_map[a["label"]]
        #                 classes.append(label_id)

        #     yield (
        #         width,
        #         height,
        #         filename,
        #         encoded_jpg,
        #         image_format,
        #         classes_text,
        #         classes,
        #     )
