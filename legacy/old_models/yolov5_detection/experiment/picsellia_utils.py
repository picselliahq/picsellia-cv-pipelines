import os
import random
import re
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import tqdm
import yaml
from picsellia.exceptions import NoDataError
from picsellia.sdk.asset import Asset, MultiAsset
from picsellia.sdk.dataset import Dataset
from picsellia.types.enums import LogType
from yolov5.utils.general import check_file, check_yaml, increment_path


def find_image_id(annotations, fname):
    for image in annotations["images"]:
        if image["file_name"] == fname:
            return image["id"]
    return None


def find_matching_annotations(dict_annotations=None, fname=None):
    img_id = find_image_id(dict_annotations, fname=fname)
    if img_id is None:
        return False, None
    ann_array = []
    for ann in dict_annotations["annotations"]:
        if ann["image_id"] == img_id:
            ann_array.append(ann)
    return True, ann_array


def to_yolo(
    assets=None,
    annotations=None,
    base_imgdir=None,
    targetdir=None,
    copy_image=True,
    split="train",
):
    """
    Simple utility function to transcribe a Picsellia Format Dataset into YOLOvX
    """
    step = split
    # Creating tree directory for YOLO
    if not os.path.isdir(targetdir):
        os.mkdir(targetdir)

    for dirname in ["images", "labels"]:
        if not os.path.isdir(os.path.join(targetdir, dirname)):
            os.mkdir(os.path.join(targetdir, dirname))

    for path in os.listdir(targetdir):
        if not os.path.isdir(os.path.join(targetdir, path, step)):
            os.mkdir(os.path.join(targetdir, path, step))

    for asset in tqdm.tqdm(assets):
        width, height = asset.width, asset.height
        success, objs = find_matching_annotations(annotations, asset.filename)

        if copy_image:
            shutil.copy(
                os.path.join(base_imgdir, asset.filename),
                os.path.join(
                    targetdir,
                    "images",
                    step,
                    asset.filename,
                ),
            )
        else:
            shutil.move(
                os.path.join(base_imgdir, asset.filename),
                os.path.join(targetdir, "images", step, asset.filename),
            )

        if success:
            label_name = f"{os.path.splitext(asset.filename)[0]}.txt"
            with open(os.path.join(targetdir, "labels", step, label_name), "w") as f:
                for a in objs:
                    x1, y1, w, h = a["bbox"]
                    category_id = a["category_id"]
                    f.write(
                        f"{category_id} {(x1 + w / 2) / width} {(y1 + h / 2) / height} {w / width} {h / height}\n"
                    )
        else:
            continue
    return


def generate_yaml(yamlname, datatargetdir, imgdir, labelmap):
    if not os.path.isdir(os.path.join(datatargetdir, "data")):
        os.mkdir(os.path.join(datatargetdir, "data"))

    dict_file = {
        "train": "{}/{}/train".format(imgdir, "images"),
        "val": "{}/{}/test".format(imgdir, "images"),
        "nc": len(labelmap),
        "names": list(labelmap.values()),
    }

    opath = f"{datatargetdir}/data/{yamlname}.yaml"
    with open(opath, "w") as file:
        yaml.dump(dict_file, file)
    return opath


def generate_data_yaml(exp, labelmap, config_path):
    cwd = os.getcwd()

    if not os.path.exists(config_path):
        os.makedirs(config_path)
    data_config_path = os.path.join(config_path, "data_config.yaml")
    n_classes = len(labelmap)
    labelmap = {int(k): v for k, v in labelmap.items()}
    ordered_labelmap = dict(sorted(OrderedDict(labelmap).items()))
    data_config = {
        "train": os.path.join(cwd, exp.png_dir, "train"),
        "val": os.path.join(cwd, exp.png_dir, "val"),
        "test": os.path.join(cwd, exp.png_dir, "test"),
        "nc": n_classes,
        "names": list(ordered_labelmap.values()),
    }
    with open(data_config_path, "w+") as f:
        yaml.dump(data_config, f, allow_unicode=True)
    return data_config_path


def edit_model_yaml(label_map, experiment_name, config_path=None):
    for path in os.listdir(config_path):
        if path.endswith("yaml"):
            ymlpath = os.path.join(config_path, path)
    path = Path(ymlpath)
    with open(ymlpath) as f:
        data = f.readlines()

    temp = re.findall(r"\d+", data[3])
    res = list(map(int, temp))

    data[3] = data[3].replace(str(res[0]), str(len(label_map)))

    if config_path is None:
        opath = (
            "."
            + ymlpath.split(".")[1]
            + "_"
            + experiment_name
            + "."
            + ymlpath.split(".")[2]
        )
    else:
        opath = (
            "./"
            + ymlpath.split(".")[0]
            + "_"
            + experiment_name
            + "."
            + ymlpath.split(".")[1]
        )
    with open(opath, "w") as f:
        for line in data:
            f.write(line)

    if config_path is None:
        tmp = opath.replace("./yolov5", ".")

    else:
        tmp = (
            ymlpath.split(".")[0] + "_" + experiment_name + "." + ymlpath.split(".")[1]
        )

    return tmp


def update_model_hyperparameters_config_file(
    hyperparameter_file_path: str, experiment_parameters: dict
) -> None:
    """
    Looking for matching key between base configuration model parameters and experiment params
    """
    if not os.path.isfile(hyperparameter_file_path):
        raise FileNotFoundError(
            f"Expected a .yaml file, got None at path: {hyperparameter_file_path}"
        )

    model_parameters = yaml.safe_load(Path(hyperparameter_file_path).read_text())

    for key, value in model_parameters.items():
        if key in experiment_parameters:
            if type(experiment_parameters[key]) not in [int, float]:
                raise TypeError(
                    f"Invalid type for parameter {key}, got {experiment_parameters[key]} expected {type(value)}"
                )
            model_parameters[key] = float(experiment_parameters[key])
            print(f"Updating param: {key} to value: {value}")

    with open(hyperparameter_file_path, "w+") as f:
        yaml.dump(model_parameters, f, allow_unicode=True)
    return


def train_test_val_split(experiment, dataset, prop, dataset_length: int = 0):
    (
        train_assets,
        test_assets,
        train_split,
        test_split,
        labels,
    ) = picsellia_train_test_split(
        dataset=dataset, prop=prop, random_seed=42, dataset_length=dataset_length
    )

    experiment.log("train-split", train_split, "bar", replace=True)
    experiment.log("test-split", test_split, "bar", replace=True)
    test_list = test_assets.items.copy()
    random.seed(42)
    random.shuffle(test_list)

    nb_asset = len(test_list) // 2
    val_data = test_list[nb_asset:]
    test_data = test_list[:nb_asset]
    val_assets = MultiAsset(dataset.connexion, dataset.id, val_data)
    test_assets = MultiAsset(dataset.connexion, dataset.id, test_data)
    return train_assets, test_assets, val_assets


def picsellia_train_test_split(
    dataset: Dataset,
    prop: float = 0.8,
    random_seed=None,
    dataset_length: int = 0,
):
    extended_assets = _fetch_extended_assets(dataset, dataset_length)
    items = _filter_assets_with_annotations(extended_assets)

    if random_seed is not None:
        random.seed(random_seed)

    train_items, eval_items = _split_assets(items, prop)

    labels = dataset.list_labels()
    label_names = {str(label.id): label.name for label in labels}

    train_assets, train_repartition = _build_assets(train_items, dataset, label_names)
    eval_assets, eval_repartition = _build_assets(eval_items, dataset, label_names)

    return (
        MultiAsset(dataset.connexion, dataset.id, train_assets),
        MultiAsset(dataset.connexion, dataset.id, eval_assets),
        train_repartition,
        eval_repartition,
        labels,
    )


def _fetch_extended_assets(dataset: Dataset, dataset_length: int) -> list:
    nb_pages = int(dataset_length / 100) + 1
    extended_items = []
    for page in range(1, nb_pages + 1):
        params = {"limit": 100, "offset": (page - 1) * 100}
        try:
            r = dataset.connexion.get(
                f"/sdk/dataset/version/{dataset.id}/assets/extended", params=params
            ).json()
            if r["count"] == 0:
                raise NoDataError("No asset with annotation found in this dataset")
            extended_items += r["items"]
        except Exception:
            pass
    return extended_items


def _filter_assets_with_annotations(extended_assets: list) -> list:
    return [item for item in extended_assets if item.get("annotations")]


def _split_assets(items: list, prop: float) -> tuple[list, list]:
    count = len(items)
    nb_assets_train = int(count * prop)
    train_eval_rep = [1] * nb_assets_train + [0] * (count - nb_assets_train)
    random.shuffle(train_eval_rep)

    train_items = []
    eval_items = []

    for item, is_train in zip(items, train_eval_rep):
        if is_train:
            train_items.append(item)
        else:
            eval_items.append(item)

    return train_items, eval_items


def _build_assets(
    items: list, dataset: Dataset, label_names: dict
) -> tuple[list, dict]:
    assets = []
    label_count: dict[str, int] = {}

    for item in items:
        annotation = item["annotations"][0]
        asset = Asset(dataset.connexion, dataset_version_id=dataset.id, data=item)
        assets.append(asset)

        label_ids: list[int] = []
        for key in ["rectangles", "classifications", "points", "polygons", "lines"]:
            label_ids.extend(shape["label_id"] for shape in annotation.get(key, []))

        for label_id in label_ids:
            try:
                label_name = label_names[label_id]
                label_count[label_name] = label_count.get(label_name, 0) + 1
            except KeyError:
                pass

    repartition = {
        "x": list(label_count.keys()),
        "y": list(label_count.values()),
    }

    return assets, repartition


def create_yolo_detection_label(exp, data_type, annotations_dict, annotations_coco):
    dataset_path = os.path.join(exp.png_dir, data_type)
    image_filenames = os.listdir(os.path.join(dataset_path, "images"))

    labels_path = os.path.join(dataset_path, "labels")

    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    for img in annotations_dict["images"]:
        img_filename = img["file_name"]
        if img_filename in image_filenames:
            create_img_label_detection(img, annotations_coco, labels_path)


def create_img_label_detection(img, annotations_coco, labels_path):
    result = []
    img_id = img["id"]
    img_filename = img["file_name"]
    w = img["width"]
    h = img["height"]
    txt_name = img_filename[:-4] + ".txt"
    annotation_ids = annotations_coco.getAnnIds(imgIds=img_id)
    anns = annotations_coco.loadAnns(annotation_ids)
    for ann in anns:
        bbox = ann["bbox"]
        yolo_bbox = coco_to_yolo_detection(bbox[0], bbox[1], bbox[2], bbox[3], w, h)
        seg_string = " ".join([str(x) for x in yolo_bbox])
        result.append(f"{ann['category_id']} {seg_string}")
    with open(os.path.join(labels_path, txt_name), "w") as f:
        f.write("\n".join(result))


def coco_to_yolo_detection(x1, y1, w, h, image_w, image_h):
    return [
        ((2 * x1 + w) / (2 * image_w)),
        ((2 * y1 + h) / (2 * image_h)),
        w / image_w,
        h / image_h,
    ]


def create_yolo_segmentation_label(exp, data_type, annotations_dict, annotations_coco):
    dataset_path = os.path.join(exp.png_dir, data_type)
    image_filenames = os.listdir(os.path.join(dataset_path, "images"))

    labels_path = os.path.join(dataset_path, "labels")

    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    for img in annotations_dict["images"]:
        img_filename = img["file_name"]
        if img_filename in image_filenames:
            create_img_label_segmentation(img, annotations_coco, labels_path)


def create_img_label_segmentation(img, annotations_coco, labels_path):
    result = []
    img_id = img["id"]
    img_filename = img["file_name"]
    w = img["width"]
    h = img["height"]
    txt_name = img_filename[:-4] + ".txt"
    annotation_ids = annotations_coco.getAnnIds(imgIds=img_id)
    anns = annotations_coco.loadAnns(annotation_ids)
    for ann in anns:
        seg = coco_to_yolo_segmentation(ann["segmentation"], w, h)
        seg_string = " ".join([str(x) for x in seg])
        result.append(f"{ann['category_id']} {seg_string}")
    with open(os.path.join(labels_path, txt_name), "w") as f:
        f.write("\n".join(result))


def countList(lst1, lst2):
    return [sub[item] for item in range(len(lst2)) for sub in [lst1, lst2]]


def coco_to_yolo_segmentation(ann, image_w, image_h):
    pair_index = np.arange(0, len(ann[0]), 2)
    impair_index = np.arange(1, len(ann[0]), 2)
    Xs = [ann[0][i] for i in pair_index]
    xs = [x / image_w for x in Xs]
    Ys = [ann[0][i] for i in impair_index]
    ys = [y / image_h for y in Ys]
    return countList(xs, ys)


def setup_hyp(
    experiment=None,
    data_yaml_path=None,
    config_path=None,
    params=None,
    label_map=None,
    cwd=None,
):
    if params is None:
        params = {}
    if label_map is None:
        label_map = []
    tmp = os.listdir(experiment.checkpoint_dir)

    for f in tmp:
        if f.endswith(".pt"):
            weight_path = os.path.join(experiment.checkpoint_dir, f)
        if f.endswith(".yaml"):
            hyp_path = os.path.join(experiment.checkpoint_dir, f)

    update_model_hyperparameters_config_file(
        hyperparameter_file_path=hyp_path, experiment_parameters=params
    )
    opt = Opt()
    opt.cwd = cwd
    opt.weights = weight_path
    opt.cfg = config_path
    opt.data = data_yaml_path
    opt.hyp = hyp_path if "hyperparams" not in params.keys() else params["hyperparams"]
    opt.epochs = 100 if "epochs" not in params.keys() else params["epochs"]
    opt.batch_size = 4 if "batch_size" not in params.keys() else params["batch_size"]
    opt.imgsz = 640 if "image_size" not in params.keys() else params["image_size"]
    opt.rect = True
    opt.resume = False
    opt.nosave = False
    opt.noval = False
    opt.noautoanchor = False
    opt.noplots = False
    opt.evolve = 300
    opt.bucket = ""
    opt.cache = "ram"
    opt.image_weights = False
    opt.device = "0" if torch.cuda.is_available() else "cpu"
    opt.multi_scale = True
    opt.single_cls = len(label_map) == 1
    opt.optimizer = "Adam"
    opt.sync_bn = False
    opt.workers = 4
    opt.project = "runs/train"
    opt.name = "exp"
    opt.exist_ok = False
    opt.quad = False
    opt.cos_lr = False
    opt.label_smoothing = 0.0
    opt.patience = 100
    opt.freeze = [0]
    opt.save_period = (
        100 if "save_period" not in params.keys() else params["save_period"]
    )
    opt.seed = 0
    opt.local_rank = -1

    opt.mask_ratio = 4
    opt.no_overlap = False

    return opt


class Opt:
    pass


def check_files(opt):
    opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
        check_file(opt.data),
        check_yaml(opt.cfg),
        check_yaml(opt.hyp),
        str(opt.weights),
        str(opt.project),
    )  # checks
    assert len(opt.cfg) or len(opt.weights), (
        "either --cfg or --weights must be specified"
    )
    if opt.evolve:
        cwd = os.getcwd()
        if opt.project == os.path.join(
            cwd, "runs/train"
        ):  # if default project name, rename to runs/evolve
            opt.project = os.path.join(cwd, "runs/evolve")
        opt.exist_ok, opt.resume = (
            opt.resume,
            False,
        )  # pass resume to exist_ok and disable resume
    if opt.name == "cfg":
        opt.name = Path(opt.cfg).stem  # use model.yaml as name
    opt.save_dir = str(
        increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    )


def find_final_run(cwd):
    runs_path = os.path.join(cwd, "runs", "train")
    dirs = os.listdir(runs_path)
    dirs.sort()
    if len(dirs) == 1:
        return os.path.join(runs_path, dirs[0])
    base = dirs[0][:7]
    truncate_dirs = [n[len(base) - 1 :] for n in dirs]
    last_run_nb = max(truncate_dirs)[-1]
    if last_run_nb == "p":
        last_run_nb = ""
    return os.path.join(runs_path, base + last_run_nb)


def get_batch_mosaics(final_run_path):
    val_batch0_labels = None
    val_batch0_pred = None
    val_batch1_labels = None
    val_batch1_pred = None
    val_batch2_labels = None
    val_batch2_pred = None
    if os.path.isfile(os.path.join(final_run_path, "val_batch0_labels.jpg")):
        val_batch0_labels = os.path.join(final_run_path, "val_batch0_labels.jpg")
    if os.path.isfile(os.path.join(final_run_path, "val_batch0_pred.jpg")):
        val_batch0_pred = os.path.join(final_run_path, "val_batch0_pred.jpg")
    if os.path.isfile(os.path.join(final_run_path, "val_batch1_labels.jpg")):
        val_batch1_labels = os.path.join(final_run_path, "val_batch1_labels.jpg")
    if os.path.isfile(os.path.join(final_run_path, "val_batch1_pred.jpg")):
        val_batch1_pred = os.path.join(final_run_path, "val_batch1_pred.jpg")
    if os.path.isfile(os.path.join(final_run_path, "val_batch2_labels.jpg")):
        val_batch2_labels = os.path.join(final_run_path, "val_batch2_labels.jpg")
    if os.path.isfile(os.path.join(final_run_path, "val_batch2_pred.jpg")):
        val_batch2_pred = os.path.join(final_run_path, "val_batch2_pred.jpg")
    return (
        val_batch0_labels,
        val_batch0_pred,
        val_batch1_labels,
        val_batch1_pred,
        val_batch2_labels,
        val_batch2_pred,
    )


def get_weights_and_config(final_run_path):
    best_weights = None
    hyp_yaml = None
    if os.path.isfile(os.path.join(final_run_path, "weights", "best.pt")):
        best_weights = os.path.join(final_run_path, "weights", "best.pt")
    if os.path.isfile(os.path.join(final_run_path, "hyp.yaml")):
        hyp_yaml = os.path.join(final_run_path, "hyp.yaml")
    return best_weights, hyp_yaml


def get_metrics_curves(final_run_path):
    curve_names = [
        "confusion_matrix",
        "F1_curve",
        "labels_correlogram",
        "labels",
        "P_curve",
        "PR_curve",
        "R_curve",
        "BoxF1_curve",
        "BoxP_curve",
        "BoxPR_curve",
        "BoxR_curve",
        "MaskF1_curve",
        "MaskP_curve",
        "MaskPR_curve",
        "MaskR_curve",
    ]
    curves = []

    for name in curve_names:
        file_ext = "jpg" if "labels" in name else "png"
        file_path = os.path.join(final_run_path, f"{name}.{file_ext}")
        if os.path.isfile(file_path):
            curves.append(file_path)
        else:
            curves.append(None)

    return tuple(curves)


def send_run_to_picsellia(experiment, cwd):
    final_run_path = find_final_run(cwd)
    best_weigths, hyp_yaml = get_weights_and_config(final_run_path)

    model_latest_path = os.path.join(final_run_path, "weights", "model.onnx")
    device = torch.device("cpu")
    model = torch.load(best_weigths, map_location=device)["model"].float()
    torch.onnx.export(
        model, torch.zeros((1, 3, 640, 640)), model_latest_path, opset_version=12
    )

    if model_latest_path is not None:
        experiment.store("model-latest", model_latest_path)
    if best_weigths is not None:
        experiment.store("checkpoint-index-latest", best_weigths)
    if hyp_yaml is not None:
        experiment.store("checkpoint-data-latest", hyp_yaml)
    for curve in get_metrics_curves(final_run_path):
        if curve is not None:
            name = curve.split("/")[-1].split(".")[0]
            experiment.log(name, curve, LogType.IMAGE)
    for batch in get_batch_mosaics(final_run_path):
        if batch is not None:
            name = batch.split("/")[-1].split(".")[0]
            experiment.log(name, batch, LogType.IMAGE)
