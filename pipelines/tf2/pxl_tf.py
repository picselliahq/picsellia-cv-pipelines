import io
import os

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


def tf_vars_generator(coco, label_map=None, annotation_type="rectangle", images_dir=None):
    if annotation_type not in ["polygon", "rectangle", "classification"]:
        raise ValueError("Please select a valid annotation_type")
    if label_map is None and annotation_type != "classification":
        raise ValueError("Provide a label_map when not working with classification")
    if images_dir is None:
        raise ValueError("images_dir is required to build image paths from COCO file_name")

    print(f"annotation type used for the variable generator: {annotation_type}")

    # Index COCO: image_id -> annotations[]
    anns_by_image_id = {}
    for ann in coco.get("annotations", []):
        anns_by_image_id.setdefault(ann["image_id"], []).append(ann)

    # Index categories: category_id -> category dict
    cats_by_id = {c["id"]: c for c in coco.get("categories", [])}

    # Helper: map category_id -> (class_id, class_name)
    # On préfère label_map si fourni (pbtxt) car TF attend souvent 1..N
    name_to_tf_id = None
    if label_map is not None:
        # label_map est supposé être dict[int,str] -> on inverse
        name_to_tf_id = {name: int(i) for i, name in label_map.items()}

    for img in coco.get("images", []):
        file_name = img.get("file_name")
        if not file_name:
            continue

        image_id = img.get("id")
        img_anns = anns_by_image_id.get(image_id, [])

        image_path = os.path.join(images_dir, file_name)
        im, encoded_jpg = process_image(image_path)
        width, height = im.size

        filename = file_name.encode("utf8")
        image_format = file_name.split(".")[-1].encode("utf8")

        # On construit une structure "image" compatible avec tes handlers existants
        # et on enrichit chaque ann avec a["label"] = {"id": tf_id, "name": class_name}
        enriched = {"annotations": []}

        for a in img_anns:
            cat = cats_by_id.get(a.get("category_id"))
            if cat is None:
                continue
            class_name = cat.get("name", "")

            # TF class id (priorité au label_map pbtxt si dispo)
            if name_to_tf_id is not None and class_name in name_to_tf_id:
                class_id = name_to_tf_id[class_name]
            else:
                # fallback: COCO category_id (moins safe si ids non 1..N)
                class_id = int(cat["id"])

            aa = dict(a)  # copy
            aa["label"] = {"id": class_id, "name": class_name}
            enriched["annotations"].append(aa)

        if annotation_type == "polygon":
            xmins, xmaxs, ymins, ymaxs, classes_text, classes, masks = (
                handle_polygon_annotations(enriched, width, height)
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
                handle_rectangle_annotations(enriched, width, height)
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

