import io
import logging
import os
import random
from typing import Dict, List

import cv2
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from models.research.object_detection import exporter_lib_v2, model_lib_v2
from models.research.object_detection.core import data_parser
from models.research.object_detection.core import standard_fields as fields
from models.research.object_detection.metrics.tf_example_parser import (
    BoundingBoxParser,
    Int64Parser,
    StringParser,
)
from models.research.object_detection.protos import pipeline_pb2
from models.research.object_detection.utils import (
    config_util,
    dataset_util,
    label_map_util,
)
from models.research.object_detection.utils import ops as utils_ops
from models.research.object_detection.utils import visualization_utils as vis_util
from PIL import Image
from tensorflow.python.summary.summary_iterator import summary_iterator

from picsellia.sdk.dataset_version import MultiAsset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logger = logging.getLogger(__name__)


def generate_label_map(classes, output_path, start=1):
    msg = ""
    for id, name in enumerate(classes, start=start):
        msg = msg + "item {\n"
        msg = msg + " id: " + str(id) + "\n"
        msg = msg + " name: '" + name + "'\n}\n\n"
    output_path = os.path.join(output_path, "label_map.pbtxt")
    with open(output_path, "w") as f:
        f.write(msg)
        f.close()
    return output_path


def sort_split(split_dict, label_names):
    label_num_list = [
        (split_dict["x"][i], split_dict["y"][i]) for i in range(len(split_dict["x"]))
    ]
    for label in label_names:
        if label not in split_dict["x"]:
            label_num_list.append((label, 0))
    label_num_dict = dict(label_num_list)
    sorted_split_dict = {
        "x": label_names,
        "y": [label_num_dict[label] for label in label_names],
    }
    return sorted_split_dict


def find_category(annotations: Dict, category_id: int):
    for category in annotations["categories"]:
        if category["id"] == category_id:
            return category
    return None


def format_coco_file(
    imgdir: str,
    annotations: Dict,
    train_assets: MultiAsset,
    eval_assets: MultiAsset,
    test_assets: MultiAsset,
):
    train_assets_fname = [e.filename for e in train_assets]
    eval_assets_fname = [e.filename for e in eval_assets]
    test_assets_fname = [e.filename for e in test_assets]
    formatted_file_train: Dict[str, List] = {"images": []}
    formatted_file_eval: Dict[str, List] = {"images": []}
    formatted_file_test: Dict[str, List] = {"images": []}
    formatted_annotations: Dict[str, List] = {}
    for shape in annotations["annotations"]:
        if shape["image_id"] in formatted_annotations.keys():
            shape["label"] = find_category(annotations, shape["category_id"])
            formatted_annotations[shape["image_id"]].append(shape)
        else:
            shape["label"] = find_category(annotations, shape["category_id"])
            formatted_annotations[shape["image_id"]] = [shape]

    for image in annotations["images"]:
        image_filename = image["file_name"]
        tmp = image.copy()
        if image["id"] in formatted_annotations:
            tmp["annotations"] = formatted_annotations[image["id"]]
            if image_filename in train_assets_fname:
                image_path = os.path.join(imgdir, "train", image_filename)
                if not os.path.isfile(image_path):
                    continue
                tmp["path"] = image_path
                formatted_file_train["images"].append(tmp)
            if image_filename in eval_assets_fname:
                image_path = os.path.join(imgdir, "eval", image_filename)
                if not os.path.isfile(image_path):
                    continue
                tmp["path"] = image_path
                formatted_file_eval["images"].append(tmp)
            if image_filename in test_assets_fname:
                image_path = os.path.join(imgdir, "test", image_filename)
                if not os.path.isfile(image_path):
                    continue
                tmp["path"] = image_path
                formatted_file_test["images"].append(tmp)
    return formatted_file_train, formatted_file_eval, formatted_file_test


def format_segmentation(shape):
    polygon = []
    xs = shape[0][::2]
    ys = shape[0][1::2]
    for x, y in list(zip(xs, ys)):
        polygon.append([x, y])
    return polygon


def create_record_files(
    train_annotations,
    eval_annotations,
    test_annotations,
    label_path,
    record_dir,
    tfExample_generator,
    annotation_type,
):
    """
    Function used to create the TFRecord files used for the training and evaluation.

    TODO: Shard files for large dataset


    Args:
        label_path: Path to the label map file.
        record_dir: Path used to write the records files.
        tfExample_generator: Use the generator from the Picsell.ia SDK by default or provide your own generator.
        annotation_type: "polygon" or "rectangle", depending on your project type. Polygon will compute the masks from your polygon-type annotations.
    """
    label_map = label_map_util.load_labelmap(label_path)
    label_map = label_map_util.get_label_map_dict(label_map)
    datasets = ["train", "eval", "test"]

    annotations_ensemble = [train_annotations, eval_annotations, test_annotations]
    for i, dataset in enumerate(datasets):
        output_path = os.path.join(record_dir, dataset + ".record")
        writer = tf.compat.v1.python_io.TFRecordWriter(output_path)
        logger.info(f"Creating record file at {output_path}")
        for variables in tfExample_generator(
            annotations_ensemble[i], label_map, annotation_type=annotation_type
        ):
            if isinstance(variables, ValueError):
                logger.error("Error", variables)
            elif annotation_type == "polygon":
                (
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
                ) = variables

                tf_example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "image/height": dataset_util.int64_feature(height),
                            "image/width": dataset_util.int64_feature(width),
                            "image/filename": dataset_util.bytes_feature(filename),
                            "image/source_id": dataset_util.bytes_feature(filename),
                            "image/encoded": dataset_util.bytes_feature(encoded_jpg),
                            "image/format": dataset_util.bytes_feature(image_format),
                            "image/object/bbox/xmin": dataset_util.float_list_feature(
                                xmins
                            ),
                            "image/object/bbox/xmax": dataset_util.float_list_feature(
                                xmaxs
                            ),
                            "image/object/bbox/ymin": dataset_util.float_list_feature(
                                ymins
                            ),
                            "image/object/bbox/ymax": dataset_util.float_list_feature(
                                ymaxs
                            ),
                            "image/object/class/text": dataset_util.bytes_list_feature(
                                classes_text
                            ),
                            "image/object/class/label": dataset_util.int64_list_feature(
                                classes
                            ),
                            "image/object/mask": dataset_util.bytes_list_feature(masks),
                        }
                    )
                )

            elif annotation_type == "rectangle":
                (
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
                ) = variables

                tf_example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "image/height": dataset_util.int64_feature(height),
                            "image/width": dataset_util.int64_feature(width),
                            "image/filename": dataset_util.bytes_feature(filename),
                            "image/source_id": dataset_util.bytes_feature(filename),
                            "image/encoded": dataset_util.bytes_feature(encoded_jpg),
                            "image/format": dataset_util.bytes_feature(image_format),
                            "image/object/bbox/xmin": dataset_util.float_list_feature(
                                xmins
                            ),
                            "image/object/bbox/xmax": dataset_util.float_list_feature(
                                xmaxs
                            ),
                            "image/object/bbox/ymin": dataset_util.float_list_feature(
                                ymins
                            ),
                            "image/object/bbox/ymax": dataset_util.float_list_feature(
                                ymaxs
                            ),
                            "image/object/class/text": dataset_util.bytes_list_feature(
                                classes_text
                            ),
                            "image/object/class/label": dataset_util.int64_list_feature(
                                classes
                            ),
                        }
                    )
                )

            writer.write(tf_example.SerializeToString())
        writer.close()
        logger.info("Successfully created the TFRecords: {}".format(output_path))


def update_num_classes(config_dict, label_map):
    """
    Update the number of classes inside the protobuf configuration dictionnary depending on the number of classes inside the label map.

        Args :
            config_dict:  A configuration dictionnary loaded from the protobuf file with config_util.get_configs_from_pipeline_file().
            label_map: Protobuf label_map loaded with label_map_util.load_labelmap()
        Raises:
            ValueError if the backbone architecture isn't known.
    """
    model_config = config_dict["model"]
    n_classes = len(label_map.item)
    meta_architecture = model_config.WhichOneof("model")
    if meta_architecture == "faster_rcnn":
        model_config.faster_rcnn.num_classes = n_classes
    elif meta_architecture == "ssd":
        model_config.ssd.num_classes = n_classes
    else:
        raise ValueError("Expected the model to be one of 'faster_rcnn' or 'ssd'.")


def check_batch_size(config_dict):
    model_config = config_dict["model"]
    meta_architecture = model_config.WhichOneof("model")
    batch_size = config_dict["train_config"].batch_size
    if meta_architecture == "faster_rcnn":
        image_resizer = model_config.faster_rcnn.image_resizer
    elif meta_architecture == "ssd":
        image_resizer = model_config.ssd.image_resizer
    else:
        raise ValueError("Unknown model type: {}".format(meta_architecture))

    if image_resizer.HasField("keep_aspect_ratio_resizer") and batch_size > 1:
        logger.warn(
            "Please be careful, your image resizer is keep_aspect_ratio_resizer and your batch size is >1."
        )
        logger.warn(
            "This mean that all your images should have the same shape. If not then set batch size to 1 or change the image resizer to a fixed_shape_resizer."
        )

    # image_resizer.HasField("fixed_shape_resizer"):


def configure_learning_rate(configs, learning_rate=None, parameters={}):
    keys = parameters.keys()

    if "lr_type" in keys:
        lr_type = parameters["lr_type"]
    else:
        lr_type = "constant"
    if "decay_steps" in keys:
        decay_steps = parameters["decay_steps"]
    else:
        decay_steps = None
    if "decay_factor" in keys:
        decay_factor = parameters["decay_factor"]
    else:
        decay_factor = None
    if "staircase" in keys:
        staircase = parameters["staircase"]
    else:
        staircase = True
    if "warmup_lr" in keys:
        warmup_lr = parameters["warmup_lr"]
    else:
        warmup_lr = None
    if "warmup_steps" in keys:
        warmup_steps = parameters["warmup_steps"]
    else:
        warmup_steps = None
    if "total_steps" in keys:
        total_steps = parameters["total_steps"]
    else:
        total_steps = None
    if "momentum" in keys:
        momentum = parameters["momentum"]
    else:
        momentum = 0.9
    if "optimizer" in keys:
        optimizer = parameters["optimizer"]
        if optimizer == "rms_prop":
            optimizer_type = "rms_prop_optimizer"
        elif optimizer == "momentum":
            optimizer_type = "momentum_optimizer"
        elif optimizer == "adam":
            optimizer_type = "adam_optimizer"
        else:
            raise TypeError("Optimizer %s is not supported." % optimizer)
    else:
        optimizer_type = configs["train_config"].optimizer.WhichOneof("optimizer")

    if optimizer_type == "rms_prop_optimizer":
        optimizer_config = configs["train_config"].optimizer.rms_prop_optimizer
    elif optimizer_type == "momentum_optimizer":
        optimizer_config = configs["train_config"].optimizer.momentum_optimizer
    elif optimizer_type == "adam_optimizer":
        optimizer_config = configs["train_config"].optimizer.adam_optimizer
    else:
        raise TypeError("Optimizer %s is not supported." % optimizer_type)

    if optimizer_type == "rms_prop_optimizer" or optimizer_type == "momentum_optimizer":
        optimizer_config.momentum_optimizer_value = min(max(0.0, momentum), 1.0)

    if lr_type == "constant":
        optimizer_config.learning_rate.constant_learning_rate.learning_rate = (
            learning_rate
        )
    elif lr_type == "exponential_decay":
        if decay_steps is not None and decay_factor is not None:
            optimizer_config.learning_rate.exponential_decay_learning_rate.initial_learning_rate = learning_rate
            optimizer_config.learning_rate.exponential_decay_learning_rate.decay_steps = decay_steps
            optimizer_config.learning_rate.exponential_decay_learning_rate.decay_factor = decay_factor
            optimizer_config.learning_rate.exponential_decay_learning_rate.staircase = (
                staircase
            )
    elif lr_type == "manual_step":
        optimizer_config.learning_rate.manual_step_learning_rate.initial_learning_rate = learning_rate
        steps = []
        lrs = []
        for k in parameters.keys():
            if k[:-1] == "step_":
                step = int(k[-1])
                if "lr_" + str(step) in parameters.keys():
                    steps.append(parameters[k])
                    lrs.append(parameters["lr_" + str(step)])
        zipped = zip(steps, lrs)
        schedules = sorted(zipped, key=lambda x: x[0])
        for i, schedule in enumerate(schedules):
            sched = optimizer_config.learning_rate.manual_step_learning_rate.schedule
            # sched.step = schedule[0]
            sched.learning_rate = schedule[1]
            optimizer_config.learning_rate.manual_step_learning_rate.schedule.append(
                sched
            )
            # optimizer_config.learning_rate.manual_step_learning_rate.schedule.step = schedule[0]
            # optimizer_config.learning_rate.manual_step_learning_rate.schedule.learning_rate = schedule[1]
    elif lr_type == "cosine_decay":
        optimizer_config.learning_rate.cosine_decay_learning_rate.learning_rate_base = (
            learning_rate
        )
        optimizer_config.learning_rate.cosine_decay_learning_rate.total_steps = (
            total_steps
        )
        optimizer_config.learning_rate.cosine_decay_learning_rate.warmup_learning_rate = warmup_lr
        optimizer_config.learning_rate.cosine_decay_learning_rate.warmup_steps = (
            warmup_steps
        )
    logger.debug(optimizer_config)


def set_image_resizer(config_dict, shape):
    """
    Update the image resizer shapes.

    Args:
        config_dict:  A configuration dictionnary loaded from the protobuf file with config_util.get_configs_from_pipeline_file().
        shape: The new shape for the image resizer.
                [max_dimension, min_dimension] for the keep_aspect_ratio_resizer (default resizer for faster_rcnn backbone).
                [width, height] for the fixed_shape_resizer (default resizer for SSD backbone)

    Raises:
        ValueError if the backbone architecture isn't known.
    """

    model_config = config_dict["model"]
    meta_architecture = model_config.WhichOneof("model")
    if meta_architecture == "faster_rcnn":
        image_resizer = model_config.faster_rcnn.image_resizer
    elif meta_architecture == "ssd":
        image_resizer = model_config.ssd.image_resizer
    else:
        raise ValueError("Unknown model type: {}".format(meta_architecture))

    if image_resizer.HasField("keep_aspect_ratio_resizer"):
        image_resizer.keep_aspect_ratio_resizer.max_dimension = shape[1]
        image_resizer.keep_aspect_ratio_resizer.min_dimension = shape[0]

    elif image_resizer.HasField("fixed_shape_resizer"):
        image_resizer.fixed_shape_resizer.height = shape[1]
        image_resizer.fixed_shape_resizer.width = shape[0]


def edit_eval_config(config_dict, annotation_type):
    """
    Update the eval_config protobuf message from a config_dict.
    Checks if the metrics_set is the right one then update the evaluation number.

    Args:
        config_dict: A configuration dictionnary loaded from the protobuf file with config_util.get_configs_from_pipeline_file().
        annotation_type: Should be either "rectangle" or "polygon". Depends on your project type.
        eval_number: The number of images you want to run your evaluation on.

    Raises:
        ValueError Wrong annotation type provided. If you didn't provide the right annotation_type
        ValueError "eval_number isn't an int". If you didn't provide a int for the eval_number.
    """

    val_config = config_dict["eval_config"]
    val_config.num_visualizations = 0
    if annotation_type == "rectangle":
        val_config.metrics_set[0] = "coco_detection_metrics"
    elif annotation_type == "polygon":
        val_config.metrics_set[0] = "coco_mask_metrics"
    else:
        raise ValueError("Wrong annotation type provided")


def update_different_paths(
    config_dict, ckpt_path, label_map_path, train_record_path, eval_record_path
):
    """
        Update the different paths required for the whole configuration.

    Args:
        config_dict: A configuration dictionnary loaded from the protobuf file with config_util.get_configs_from_pipeline_file().
        ckpt_path: Path to your checkpoint.
        label_map_path: Path to your label map.
        train_record_path: Path to your train record file.
        eval_record_path: Path to your eval record file.

    """
    config_dict["train_config"].fine_tune_checkpoint = ckpt_path
    config_util._update_label_map_path(config_dict, label_map_path)
    config_util._update_tf_record_input_path(
        config_dict["train_input_config"], train_record_path
    )
    config_util._update_tf_record_input_path(
        config_dict["eval_input_config"], eval_record_path
    )


def edit_masks(config_dict, mask_type="PNG_MASKS"):
    """
    Update the configuration to take into consideration the right mask_type. By default we record mask as "PNG_MASKS".

    Args:
        config_dict: A configuration dictionnary loaded from the protobuf file with config_util.get_configs_from_pipeline_file().
        mask_type: String name to identify mask type, either "PNG_MASKS" or "NUMERICAL_MASKS"
    Raises:
        ValueError if the mask type isn't known.
    """

    config_dict["train_input_config"].load_instance_masks = True
    config_dict["eval_input_config"].load_instance_masks = True
    if mask_type == "PNG_MASKS":
        config_dict["train_input_config"].mask_type = 2
        config_dict["eval_input_config"].mask_type = 2
    elif mask_type == "NUMERICAL_MASKS":
        config_dict["train_input_config"].mask_type = 1
        config_dict["eval_input_config"].mask_type = 1
    else:
        raise ValueError("Wrong Mask type provided")


def set_variable_loader(config_dict, incremental_or_transfer, FromSratch=False):
    """
        Choose the training type. If incremental then all variables from the checkpoint are loaded, used to resume a training.
    Args:
        config_dict: A configuration dictionnary loaded from the protobuf file with config_util.get_configs_from_pipeline_file().
        incremental_or_transfer: String name to identify use case "transfer" of "incremental".
    Raises:
        ValueError
    """
    if not FromSratch:
        config_dict["train_config"].from_detection_checkpoint = True


def edit_config(
    model_selected,
    input_config_dir,
    output_config_dir,
    num_steps,
    label_map_path,
    train_record_path,
    eval_record_path,
    annotation_type,
    batch_size=None,
    learning_rate=None,
    parameters={},
    resizer_size=None,
    incremental_or_transfer="transfer",
):
    """
    Wrapper to edit the essential values inside the base configuration protobuf file provided with an object-detection/segmentation checkpoint.
    This configuration file is what will entirely define your model, pre-processing, training, evaluation etc. It is the most important file of a model with the checkpoint file and should never be deleted.
    This is why it is saved in almost every directory where you did something to keep redondancy but also to be sure to have the right config file used at this moment.
    For advanced users, if you want to dwell deep inside the configuration file you should read the proto definitions inside the proto directory of the object-detection API.

    Args:
        Required:
            model_selected: The checkpoint you want to resume from.
            config_output_dir: The path where you want to save your edited protobuf configuration file.
            num_steps: The number of steps you want to train on.
            label_map_path: The path to your label_map.pbtxt file.
            record_dir: The path to the directory where your TFRecord files are saved.
            eval_number: The number of images you want to evaluate on.
            annotation_type: Should be either "rectangle" or "polygon", depending on how you annotated your images.

        Optional:
            batch_size: The batch size you want to use. If not provided it will use the previous one.
            learning_rate: The learning rate you want to use for the training. If not provided it will use the previous one.
                            Please see config_utils.update_initial_learning_rate() inside the object_detection folder for indepth details on what happens when updating it.
            resizer_size: The shape used to update your image resizer. Please see set_image_resizer() for more details on this. If not provided it will use the previous one.

    """

    file_list = os.listdir(model_selected)
    ckpt_ids = []
    for p in file_list:
        if "index" in p:
            if "-" in p:
                ckpt_ids.append(int(p.split("-")[1].split(".")[0]))
    if len(ckpt_ids) > 0:
        ckpt_path = os.path.join(model_selected, "ckpt-{}".format(str(max(ckpt_ids))))

    else:
        ckpt_path = os.path.join(model_selected, "ckpt")

    configs = config_util.get_configs_from_pipeline_file(
        os.path.join(input_config_dir, "pipeline.config")
    )
    label_map = label_map_util.load_labelmap(label_map_path)
    # num_steps += configs["train_config"].num_steps
    config_util._update_train_steps(configs, num_steps)
    update_different_paths(
        configs,
        ckpt_path=ckpt_path,
        label_map_path=label_map_path,
        train_record_path=train_record_path,
        eval_record_path=eval_record_path,
    )

    if learning_rate is not None:
        configure_learning_rate(configs, learning_rate, parameters)
    if batch_size is not None:
        config_util._update_batch_size(configs, batch_size)
    check_batch_size(configs)

    if annotation_type == "polygon":
        edit_masks(configs, mask_type="PNG_MASKS")

    if resizer_size is not None:
        set_image_resizer(configs, resizer_size)

    if incremental_or_transfer is not None:
        set_variable_loader(configs, incremental_or_transfer)

    edit_eval_config(configs, annotation_type)
    update_num_classes(configs, label_map)
    config_proto = config_util.create_pipeline_proto_from_configs(configs)
    config_util._update_batch_size(configs, batch_size)
    config_util.save_pipeline_config(config_proto, directory=output_config_dir)
    logger.info(f"Configuration successfully edited and saved at {output_config_dir}")


def train(
    model_dir="",
    config_dir="",
    train_steps=None,
    use_tpu=False,
    checkpoint_every_n=100,
    record_summaries=True,
    log_real_time=None,
    evaluate_fn=None,
    log_metrics=None,
):
    if config_dir.endswith(".config"):
        if not os.path.isfile(config_dir):
            raise FileNotFoundError("No config file found at {}".format(config_dir))
        else:
            config = config_dir
    else:
        if os.path.isdir(config_dir):
            files = os.listdir(config_dir)
            file_found = False
            for f in files:
                if not file_found:
                    if f.endswith(".config"):
                        config = os.path.join(config_dir, f)
                        file_found = True
    if not file_found:
        raise FileNotFoundError(
            "No config file found in this directory {}".format(config_dir)
        )

    tf.config.set_soft_device_placement(True)
    strategy = tf.compat.v2.distribute.MirroredStrategy()

    with strategy.scope():
        model_lib_v2.train_loop(
            pipeline_config_path=config,
            model_dir=model_dir,
            train_steps=train_steps,
            use_tpu=use_tpu,
            checkpoint_every_n=checkpoint_every_n,
            record_summaries=record_summaries,
            picsellia_experiment=log_real_time,
            evaluate_fn=evaluate_fn,
            log_metrics=log_metrics,
        )


def evaluate(metrics_dir="", config="", ckpt_dir="", train_steps=None):
    """
    Function used to evaluate your trained model.

    Args:
        Required:
            eval_dir: The directory where the tfevent file will be saved.
            config: The protobuf configuration file or directory.
            checkpoint_dir: The directory where the checkpoint you want to evaluate is.

        Optional:
            eval_training_data: Is set to True the evaluation will be run on the training dataset.

    Returns:
        A dictionnary of metrics ready to be sent to the picsell.ia platform.
    """
    if config.endswith(".config"):
        if not os.path.isfile(config):
            raise FileNotFoundError("No config file found at {}".format(config))
    else:
        if os.path.isdir(config):
            files = os.listdir(config)
            file_found = False
            for f in files:
                if not file_found:
                    if f.endswith(".config"):
                        config = os.path.join(config, f)
                        file_found = True
    if not file_found:
        raise FileNotFoundError(
            "No config file found in this directory {}".format(config)
        )

    tf.config.set_soft_device_placement(True)
    model_lib_v2.eval_continuously(
        pipeline_config_path=config,
        model_dir=metrics_dir,
        train_steps=train_steps,
        sample_1_of_n_eval_examples=1,
        sample_1_of_n_eval_on_train_examples=0,
        checkpoint_dir=ckpt_dir,
        wait_interval=4,
        timeout=5,
    )


def tf_events_to_dict(dir_path, type=""):
    """Get a dictionnary of scalars from the tfevent inside the training directory.

    Args:
        path: The path to the directory where a tfevent file is saved or the path to the file.

    Returns:
        A dictionnary of scalars logs.
    """
    file_found = False
    log_dict = {}
    if dir_path.startswith("events.out"):
        if not os.path.isfile(dir_path):
            raise FileNotFoundError("No tfEvent file found at {}".format(dir_path))
    else:
        if os.path.isdir(dir_path):
            files = os.listdir(dir_path)
            file_found = False
            last_num_val = 0
            for f in files:
                if f.startswith("events.out"):
                    list_split_f = f.split(".")
                    if int(list_split_f[-2]) >= last_num_val:
                        last_num_val = int(list_split_f[-2])
                        file_path = os.path.join(dir_path, f)
                        file_found = True
    if not file_found:
        raise FileNotFoundError(
            "No tfEvent file found in this directory {}".format(dir_path)
        )
    for summary in summary_iterator(file_path):
        for v in summary.summary.value:
            if "image" not in v.tag:
                if v.tag in log_dict.keys():
                    decoded = tf.compat.v1.decode_raw(
                        v.tensor.tensor_content, tf.float32
                    )
                    log_dict[v.tag]["steps"].append(
                        str(len(log_dict[v.tag]["steps"]) + 1)
                    )
                    log_dict[v.tag]["values"].append(
                        str(tf.cast(decoded, tf.float32).numpy()[0])
                    )
                else:
                    decoded = tf.compat.v1.decode_raw(
                        v.tensor.tensor_content, tf.float32
                    )
                    if type == "train":
                        scalar_dict = {
                            "steps": [0],
                            "values": [str(tf.cast(decoded, tf.float32).numpy()[0])],
                        }
                        log_dict[v.tag] = scalar_dict
                    if type == "eval":
                        log_dict[v.tag] = str(tf.cast(decoded, tf.float32).numpy()[0])

    return log_dict


def log_metrics(experiment, tf_metrics_dir, global_step, metrics_type):
    if metrics_type == "train":
        metrics = tf_events_to_dict(
            dir_path=os.path.join(tf_metrics_dir, "train"), type=metrics_type
        )
        log_type = "Train"
        for metric_name, value in metrics.items():
            data = {
                "steps": [float(global_step)],
                "values": [float(value["values"][-1])],
            }
            experiment.log(log_type + "/" + metric_name, data, "line")
    elif metrics_type == "eval":
        metrics = tf_events_to_dict(
            dir_path=os.path.join(tf_metrics_dir, "eval"), type=metrics_type
        )
        log_type = "Validation"
        for metric_name, value in metrics.items():
            data = {"steps": [float(global_step)], "values": [float(value)]}
            experiment.log(log_type + "/" + metric_name, data, "line")


def export_graph(
    ckpt_dir,
    exported_model_dir,
    config_dir,
    config_override="",
    input_type="image_tensor",
    use_side_inputs=False,
    side_input_shapes="",
    side_input_types="",
    side_input_names="",
):
    """Export your checkpoint to a saved_model.pb file
    Args:
        Required:
            ckpt_dir: The directory where your checkpoint to export is located.
            exported_model_dir: The directory where you want to save your model.
            pipeline_çonfig_path: The directory where you protobuf configuration is located.

    """
    pipeline_config_path = os.path.join(config_dir, "pipeline.config")
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(pipeline_config_path, "r") as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge(config_override, pipeline_config)

    exporter_lib_v2.export_inference_graph(
        input_type=input_type,
        pipeline_config=pipeline_config,
        trained_checkpoint_dir=ckpt_dir,
        output_directory=exported_model_dir,
        use_side_inputs=use_side_inputs,
        side_input_shapes=side_input_shapes,
        side_input_types=side_input_types,
        side_input_names=side_input_names,
    )


def infer(
    path,
    exported_model_dir,
    label_map_path,
    results_dir,
    disp=True,
    num_infer=5,
    min_score_thresh=0.7,
    from_tfrecords=False,
):
    """Use your exported model to infer on a path list of images.

    Args:
        Required:
            path_list: A list of images paths to infer on. Or the path to the TFRecord directory if infering from TFRecords files.
            exported_model_dir: The path used to saved your model.
            label_mapt_path: The path to your label_map file.
            results_dir: The directory where you want to save your infered images.

        Optional:
            num_infer: The number of images you want to infer on.
            min_score_tresh: The minimal confidence treshold to keep the detection.

    """
    saved_model_path = os.path.join(exported_model_dir, "saved_model")
    model = tf.saved_model.load(saved_model_path)
    if not from_tfrecords:
        path_list = path
        random.shuffle(path_list)

    elif from_tfrecords:
        eval_path = os.path.join(path, "eval.record")
        path_list = []
        for example in tf.compat.v1.python_io.tf_record_iterator(eval_path):
            tf_examp = tf.train.Example.FromString(example)
            for key in tf_examp.features.feature:
                if key == "image/filename":
                    path_list.append(
                        tf_examp.features.feature[key]
                        .bytes_list.value[0]
                        .decode("utf-8")
                    )
    path_list = path_list[:num_infer]
    category_index = label_map_util.create_category_index_from_labelmap(label_map_path)
    counter = 0
    for img_path in path_list:
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = np.expand_dims(img, 0)
            result = model(img_tensor)
            logger.debug(result["detection_classes"][0])
            if "detection_masks" in result:
                # we need to convert np.arrays to tensors
                detection_masks = tf.convert_to_tensor(result["detection_masks"][0])
                detection_boxes = tf.convert_to_tensor(result["detection_boxes"][0])

                # Reframe the the bbox mask to the image size.
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, img.shape[1], img.shape[2]
                )
                detection_masks_reframed = tf.cast(
                    detection_masks_reframed > 0.5, tf.uint8
                )
                result["detection_masks_reframed"] = detection_masks_reframed.numpy()

            vis_util.visualize_boxes_and_labels_on_image_array(
                img,
                (result["detection_boxes"][0]).numpy(),
                (result["detection_classes"][0]).numpy().astype(int),
                (result["detection_scores"][0]).numpy(),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=0.30,
                agnostic_mode=False,
                instance_masks=result.get("detection_masks_reframed", None),
                line_thickness=8,
            )

            img_name = img_path.split("/")[-1]
            Image.fromarray(img).save(os.path.join(results_dir, img_name))

        except Exception:
            counter += 1
            continue
    if counter > 0:
        logger.info(f"{counter} weren't infered")


def run():
    pass


FLAGS = tf.compat.v1.flags.FLAGS


class CustomParser(data_parser.DataToNumpyParser):
    """Tensorflow Example proto parser."""

    def __init__(self):
        self.items_to_handlers = {
            fields.InputDataFields.image: StringParser(
                fields.TfExampleFields.image_encoded
            ),
            fields.InputDataFields.groundtruth_boxes: BoundingBoxParser(
                fields.TfExampleFields.object_bbox_xmin,
                fields.TfExampleFields.object_bbox_ymin,
                fields.TfExampleFields.object_bbox_xmax,
                fields.TfExampleFields.object_bbox_ymax,
            ),
            fields.InputDataFields.groundtruth_classes: Int64Parser(
                fields.TfExampleFields.object_class_label
            ),
        }
        # Optional
        self.filename = StringParser(fields.TfExampleFields.filename)

    def parse(self, tf_example):
        """
        Parses tensorflow example and returns a tensor dictionary.
        Args:
            tf_example: a tf.Example object.
        Returns:
            A dictionary of the following numpy arrays:
            image               - string containing input image.
            filename            - string containing input filename (optional, None if not specified)
            groundtruth_boxes   - a numpy array containing groundtruth boxes.
            groundtruth_classes - a numpy array containing groundtruth classes.
        """
        results_dict = {}
        parsed = True
        for key, parser in self.items_to_handlers.items():
            results_dict[key] = parser.parse(tf_example)
            parsed &= results_dict[key] is not None

        # TODO: need to test
        filename = self.filename.parse(tf_example)
        results_dict["filename"] = filename  # could be None

        return results_dict if parsed else None


def compute_iou(groundtruth_box, detection_box):
    g_ymin, g_xmin, g_ymax, g_xmax = tuple(groundtruth_box.tolist())
    d_ymin, d_xmin, d_ymax, d_xmax = tuple(detection_box.tolist())

    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

    return intersection / float(boxAArea + boxBArea - intersection)


def get_confusion_matrix(
    input_tfrecord_path,
    model,
    labelmap,
    iou_thresh=0.5,
    confidence_thresh=0.5,
    draw_option=False,
    draw_save_path="",
):
    """
    Creates input dataset from tfrecord, runs detection model, compares detection results with ground truth
    Args:
        input_tfrecord_path: path of input tfrecord file
        model: path of detection model .pb file
        categories: ordered array of class IDs
        draw_option: whether or not to visualize and save detections and ground truth boxes
        draw_save_path: where to save visualizations if draw_option is true
    """
    IOU_THRESHOLD = iou_thresh
    CONFIDENCE_THRESHOLD = confidence_thresh
    data_parser = CustomParser()
    categories = [lab for lab in labelmap.keys()]
    labels = [lab for lab in labelmap.values()]
    confusion_matrix = np.zeros(shape=(len(categories) + 1, len(categories) + 1))
    filename_matrix = [
        [[] for i in range(len(categories) + 1)] for j in range(len(categories) + 1)
    ]
    # Create dataset from records
    input_dataset = tf.data.TFRecordDataset(input_tfrecord_path)
    # model = tf.saved_model.load(model_path)
    # with progressbar.ProgressBar(max_value=progressbar.UnknownLength) as bar:
    results = []
    model = tf.saved_model.load(model)
    total_objects = np.zeros(len(categories))
    for image_index, record in enumerate(input_dataset):
        if image_index < 3:
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            decoded_dict = data_parser.parse(example)
            if decoded_dict:
                image = decoded_dict[fields.InputDataFields.image]
                input_tensor = np.expand_dims(Image.open(io.BytesIO(image)), axis=0)
                groundtruth_boxes = decoded_dict[
                    fields.InputDataFields.groundtruth_boxes
                ]
                groundtruth_classes = decoded_dict[
                    fields.InputDataFields.groundtruth_classes
                ].astype("uint8")
                detections = model(input_tensor)  # Run model inference
                detection_scores = detections["detection_scores"][0].numpy()
                detection_boxes = detections["detection_boxes"][0].numpy()[
                    detection_scores >= CONFIDENCE_THRESHOLD
                ]
                detection_classes = (
                    detections["detection_classes"][0]
                    .numpy()[detection_scores >= CONFIDENCE_THRESHOLD]
                    .astype("uint8")
                )
                filename = decoded_dict[fields.InputDataFields.filename]
                filename = (
                    filename.decode("UTF-8")
                    if filename is not None
                    else f"image-{image_index}.png"
                )
                matches = []
                image_index += 1
                gt_boxes = []
                pred_boxes = []
                for i, groundtruth_box in enumerate(groundtruth_boxes):
                    for j, detection_box in enumerate(detection_boxes):
                        iou = compute_iou(groundtruth_box, detection_box)
                        if iou > IOU_THRESHOLD:
                            matches.append([i, j, iou])
                            gt_boxes.append(groundtruth_boxes)
                            pred_boxes.append(detection_boxes)
                matches = np.array(matches)
                if matches.shape[0] > 0:
                    # Sort list of matches by descending IOU so we can remove duplicate detections
                    # while keeping the highest IOU entry.
                    matches = matches[matches[:, 2].argsort()[::-1][: len(matches)]]

                    # Remove duplicate detections from the list.
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

                    # Sort the list again by descending IOU. Removing duplicates doesn't preserve
                    # our previous sort.
                    matches = matches[matches[:, 2].argsort()[::-1][: len(matches)]]

                    # Remove duplicate ground truths from the list.
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                nb_objects = np.zeros(len(categories))
                nb_preds = np.zeros(len(categories))
                FN = np.zeros(len(categories))
                FP = np.zeros(len(categories))
                TP = np.zeros(len(categories))
                for i in range(len(groundtruth_boxes)):
                    if (
                        matches.shape[0] > 0
                        and matches[matches[:, 0] == i].shape[0] == 1
                    ):
                        confusion_matrix[groundtruth_classes[i] - 1][
                            int(
                                detection_classes[
                                    int(matches[matches[:, 0] == i, 1][0])
                                ]
                                - 1
                            )
                        ] += 1
                        if (
                            filename
                            not in filename_matrix[groundtruth_classes[i] - 1][
                                int(
                                    detection_classes[
                                        int(matches[matches[:, 0] == i, 1][0])
                                    ]
                                    - 1
                                )
                            ]
                        ):
                            filename_matrix[groundtruth_classes[i] - 1][
                                int(
                                    detection_classes[
                                        int(matches[matches[:, 0] == i, 1][0])
                                    ]
                                    - 1
                                )
                            ].append(filename)
                        if (
                            int(
                                detection_classes[
                                    int(matches[matches[:, 0] == i, 1][0])
                                ]
                                - 1
                            )
                            == groundtruth_classes[i] - 1
                        ):
                            TP[groundtruth_classes[i] - 1] += 1
                        else:
                            FP[
                                int(
                                    detection_classes[
                                        int(matches[matches[:, 0] == i, 1][0])
                                    ]
                                    - 1
                                )
                            ] += 1
                    else:
                        confusion_matrix[groundtruth_classes[i] - 1][
                            confusion_matrix.shape[1] - 1
                        ] += 1
                        if (
                            filename
                            not in filename_matrix[groundtruth_classes[i] - 1][
                                confusion_matrix.shape[1] - 1
                            ]
                        ):
                            filename_matrix[groundtruth_classes[i] - 1][
                                confusion_matrix.shape[1] - 1
                            ].append(filename)
                        FN[groundtruth_classes[i] - 1] += 1
                    total_objects[groundtruth_classes[i] - 1] += 1
                    nb_objects[groundtruth_classes[i] - 1] += 1

                for i in range(len(detection_boxes)):
                    if (
                        matches.shape[0] > 0
                        and matches[matches[:, 1] == i].shape[0] == 0
                    ):
                        confusion_matrix[confusion_matrix.shape[0] - 1][
                            int(detection_classes[i] - 1)
                        ] += 1
                        if (
                            filename
                            not in filename_matrix[confusion_matrix.shape[0] - 1][
                                int(detection_classes[i] - 1)
                            ]
                        ):
                            filename_matrix[confusion_matrix.shape[0] - 1][
                                int(detection_classes[i] - 1)
                            ].append(filename)
                        FP[detection_classes[i] - 1] += 1
                    nb_preds[detection_classes[i] - 1] += 1

                image_dict = {
                    "filename": filename.split("/")[-1],
                    "nb_objects": nb_objects.tolist(),
                    "nb_preds": nb_preds.tolist(),
                    "TP": TP.tolist(),
                    "FP": FP.tolist(),
                    "FN": FN.tolist(),
                    "bbox": detection_boxes.tolist(),
                    "classes": detection_classes.tolist(),
                }
                results.append(image_dict)
            # if draw_option:
            #     draw(filename, draw_save_path, input_tensor[0], categories,
            #         groundtruth_boxes, groundtruth_classes, detection_boxes, detection_classes, detection_scores)
            else:
                logger.warn(f"Skipping image {image_index}")

    logger.info(f"Processed {image_index + 1} images")
    evaluation = {
        "categories": categories,
        "labels": labels,
        "total_objects": total_objects.tolist(),
        "per_image_eval": results,
        "filename_matrix": filename_matrix,
        "metrics": {
            "iou": IOU_THRESHOLD,
            "confidence": CONFIDENCE_THRESHOLD,
        },
    }
    return confusion_matrix, evaluation
