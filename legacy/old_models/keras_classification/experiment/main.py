# type: ignore
import logging
import os
import random

import keras
import numpy as np
import tensorflow as tf
import tqdm
import utils
from classification_models.keras import Classifiers
from picsellia.sdk.experiment import Experiment
from picsellia.types.enums import AnnotationFileType, InferenceType, LogType
from pycocotools.coco import COCO
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import (
    _move_files_in_class_directories,
    get_experiment,
    get_train_test_eval_datasets_from_experiment,
    order_repartition_according_labelmap,
)

os.environ["PICSELLIA_SDK_CUSTOM_LOGGING"] = "True"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"

logging.getLogger("picsellia").setLevel(logging.INFO)

experiment: Experiment = get_experiment()

model_architecture = os.environ["architecture"]

experiment.start_logging_chapter("Dowloading test_files")

experiment.download_artifacts(with_tree=True)

artifact = experiment.get_artifact("keras-model")
model_name = artifact.filename
model_path = os.path.join(experiment.checkpoint_dir, model_name)

os.rename(os.path.join(experiment.base_dir, model_name), model_path)

is_split, train_ds, test_ds, eval_ds = get_train_test_eval_datasets_from_experiment(
    experiment
)
if not is_split:
    dataset = train_ds
    eval_ds = train_ds
    print("Downloading annotation COCO file ...")
    annotation_path = dataset.export_annotation_file(
        AnnotationFileType.COCO, experiment.base_dir
    )
    print("Downloading annotation COCO file ... OK")

    coco_train = COCO(annotation_path)
    coco_test = coco_train
    coco_eval = coco_train
    (
        train_assets,
        test_assets,
        eval_assets,
        count_train,
        count_test,
        count_eval,
        _,
    ) = dataset.train_test_val_split()
    labelmap = {}
    for x in coco_train.cats:
        labelmap[str(x)] = coco_train.cats[x]["name"]
    experiment.log(
        "train-split",
        order_repartition_according_labelmap(labelmap, count_train),
        "bar",
        replace=True,
    )
    experiment.log(
        "test-split",
        order_repartition_according_labelmap(labelmap, count_test),
        "bar",
        replace=True,
    )
    experiment.log(
        "eval-split",
        order_repartition_according_labelmap(labelmap, count_eval),
        "bar",
        replace=True,
    )

    dataset_labels = {label.name: label for label in dataset.list_labels()}


else:
    print("Downloading annotation COCO file ...")
    train_annotation_path = train_ds.export_annotation_file(
        AnnotationFileType.COCO, experiment.base_dir
    )
    print("Downloading annotation COCO file ... OK")
    print("Downloading annotation COCO file ...")
    test_annotation_path = test_ds.export_annotation_file(
        AnnotationFileType.COCO, experiment.base_dir
    )
    print("Downloading annotation COCO file ... OK")
    print("Downloading annotation COCO file ...")
    eval_annotation_path = eval_ds.export_annotation_file(
        AnnotationFileType.COCO, experiment.base_dir
    )
    print("Downloading annotation COCO file ... OK")

    coco_train = COCO(train_annotation_path)
    coco_test = COCO(test_annotation_path)
    coco_eval = COCO(eval_annotation_path)

    train_assets = train_ds.list_assets()
    test_assets = test_ds.list_assets()
    eval_assets = eval_ds.list_assets()

    dataset_labels = {label.name: label for label in eval_ds.list_labels()}

random.shuffle(train_assets)
random.shuffle(test_assets)
random.shuffle(eval_assets)
train_assets.download(target_path=os.path.join(experiment.png_dir, "train"))
test_assets.download(target_path=os.path.join(experiment.png_dir, "test"))
eval_assets.download(target_path=os.path.join(experiment.png_dir, "eval"))

_move_files_in_class_directories(
    coco=coco_train, base_imdir=os.path.join(experiment.png_dir, "train")
)
_move_files_in_class_directories(
    coco=coco_test, base_imdir=os.path.join(experiment.png_dir, "test")
)
_move_files_in_class_directories(
    coco=coco_eval, base_imdir=os.path.join(experiment.png_dir, "eval")
)

labelmap = {}
for x in coco_train.cats:
    labelmap[str(x)] = coco_train.cats[x]["name"]

n_classes = len(labelmap)
experiment.log("labelmap", labelmap, "labelmap", replace=True)

parameters = experiment.get_log(name="parameters").data
random_seed = parameters.get("random_seed", 12)
target_width = int(parameters.get("resized_width", 200))
target_height = int(parameters.get("resized_height", 200))
EPOCHS = parameters.get("epochs", 100)
INITIAL_LR = parameters.get("learning_rate", 0.0005)
target_size = (target_width, target_height)
target_input = (target_width, target_height, 3)
batch_size = int(parameters.get("batch_size", 128))
target_size = (target_width, target_height)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    dtype="float32",
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0, dtype="float32")
eval_datagen = test_datagen

train_generator = train_datagen.flow_from_directory(
    os.path.join(experiment.png_dir, "train"),
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
    seed=random_seed,
    shuffle=True,
)
test_generator = test_datagen.flow_from_directory(
    os.path.join(experiment.png_dir, "test"),
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
    seed=random_seed,
    shuffle=True,
)
eval_generator = eval_datagen.flow_from_directory(
    os.path.join(experiment.png_dir, "eval"),
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
    seed=random_seed,
    shuffle=True,
)

experiment.start_logging_chapter("Init Model")

Architecture, preprocess_input = Classifiers.get(model_architecture)

base_model = Architecture(input_shape=target_input, include_top=False, weights=None)

try:
    base_model.load_weights(os.path.join(experiment.checkpoint_dir, model_name))
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])
except ValueError as e:
    print(str(e))
    print("Adding an extra Average Pooling Layer .. OK")
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])
    model.load_weights(os.path.join(experiment.checkpoint_dir, model_name))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=parameters["learning_rate"]),
    loss="categorical_crossentropy",
    metrics=["accuracy", utils.f1_micro],
)

experiment.start_logging_chapter("Training")

class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes,
)
history = model.fit(
    train_generator,
    batch_size=batch_size,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[
        utils.Metrics(
            val_data=test_generator, batch_size=batch_size, experiment=experiment
        )
    ],
    verbose=1,
    class_weight=dict(enumerate(class_weights)),
)
for k, v in history.history.items():
    try:
        experiment.log(k, list(map(float, v)), LogType.LINE)
    except Exception:
        print(f"can't send {v}")

model.save(os.path.join(experiment.exported_model_dir, "model.h5"))
model.save_weights(os.path.join(experiment.exported_model_dir, "cp.ckpt"))
tf.saved_model.save(model, os.path.join(experiment.exported_model_dir, "saved_model"))
experiment.store(
    "model-latest",
    os.path.join(experiment.exported_model_dir, "saved_model"),
    do_zip=True,
)
experiment.store(
    name="keras-model", path=os.path.join(experiment.exported_model_dir, "model.h5")
)
experiment.store(
    name="checkpoint-index",
    path=os.path.join(experiment.exported_model_dir, "cp.ckpt.index"),
)
experiment.store(
    name="checkpoint-data",
    path=os.path.join(experiment.exported_model_dir, "cp.ckpt.data-00000-of-00001"),
)

experiment.start_logging_chapter("Evaluation")

predictions = model.predict(eval_generator)

eval_accuracy = accuracy_score(eval_generator.classes, predictions.argmax(axis=1))

experiment.log(name="eval_accuracy", data=eval_accuracy, type=LogType.VALUE)

cm = confusion_matrix(eval_generator.classes, predictions.argmax(axis=1))

confusion = {"categories": list(labelmap.values()), "values": cm.tolist()}
log = experiment.log(name="confusion", data=confusion, type=LogType.HEATMAP)

for i, pred in enumerate(tqdm.tqdm(predictions)):
    asset_filename = eval_generator.filenames[i].split("/")[1]
    try:
        asset = eval_ds.find_asset(filename=asset_filename)
    except Exception as e:
        print(e)
    experiment.add_evaluation(
        asset=asset,
        classifications=[
            (dataset_labels[labelmap[str(np.argmax(pred))]], float(max(pred)))
        ],
    )
    print(f"Asset: {asset_filename} evaluated.")
experiment.compute_evaluations_metrics(InferenceType.CLASSIFICATION)
