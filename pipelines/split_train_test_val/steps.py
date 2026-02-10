import json

from picsellia.types.enums import InferenceType
from picsellia_cv_engine.core import CocoDataset, Model
from picsellia_cv_engine.core.contexts import PicselliaDatasetProcessingContext
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step

from utils.processing import process_images

from picsellia.exceptions import PicselliaError


@step
def length_dataset_version_sanity_check()-> bool:
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    print(f"Verification of the DatasetVersion length")
    assert len(context.input_dataset_version.list_assets()) >= 3, PicselliaError(f"The DatasetVersion has less than 3 assets.")
    return True

@step
def parameters_sanity_chek()-> bool:
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    parameters = context.processing_parameters.to_dict()
    print(f"Verification of the parameters")
    assert (parameters.get('ratio_train')+parameters.get('ratio_val')+parameters.get('ratio_test') == 1, PicselliaError(f"The sum of the three ratios is not 1 but {parameters.get('ratio_train')+parameters.get('ratio_val')+parameters.get('ratio_test')}"))
    assert (parameters.get('ratio_train') != 0, PicselliaError(f"The train dataset cannot be empty"))
    return True


@step
def create_empty_annotation()->bool:
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    parameters = context.processing_parameters.to_dict()

    dataset_version = context.input_dataset_version
    if parameters.get('embed_asset_without_annotation')=="True":
        print(f"Creating empty annotations for assets without existing annotations")
        assert dataset_version!=InferenceType.NOT_CONFIGURED, PicselliaError(f"The DatasetVersion type should be configured to create empty annotations")
        empty_assets = [asset for asset in dataset_version.list_assets() if len(asset.list_annotations())==0]
        for empty_asset in empty_assets:
            empty_asset.create_annotation()
        print(f"{str(len(empty_assets))} empty annotations created")


@step   
def split_and_tag_data(self)-> None:
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    parameters = context.processing_parameters.to_dict()
    dataset_version = context.input_dataset_version
    dataset_version_name = dataset_version.version
    
    print(f"Start splitting")
    if float(parameters.get('ratio_val')) != 0 and float(parameters.get('ratio_test')) != 0:
        train_assets, test_assets, val_assets, _, _, _, _ = dataset_version.train_test_val_split(ratios=[float(parameters.get('ratio_train')),float(parameters.get('ratio_test')),float(parameters.get('ratio_val'))])
        try:
            train_assets.add_tags(dataset_version.create_asset_tag("train"))
        except:
            pass
        _,job_train = dataset_version.fork(version=dataset_version_name+"_train",assets=train_assets, type= dataset_version.type, with_tags = False ,with_labels = True, with_annotations = True, wait=False)
        job_train.wait_for_done(blocking_time_increment=5.0, attempts=360)
        try:
            val_assets.add_tags(dataset_version.create_asset_tag("val"))
        except:
            pass
        _,job_val = dataset_version.fork(version=dataset_version_name+"_val",assets=val_assets, type= dataset_version.type, with_tags = False ,with_labels = True, with_annotations = True, wait=False)
        job_val.wait_for_done(blocking_time_increment=5.0, attempts=360)
        try:
            test_assets.add_tags(dataset_version.create_asset_tag("test"))
        except:
            pass
        _,job_test = dataset_version.fork(version=dataset_version_name+"_test",assets=test_assets, type= dataset_version.type, with_tags = False ,with_labels = True, with_annotations = True, wait=False)
        job_test.wait_for_done(blocking_time_increment=5.0, attempts=360)
            

    elif float(parameters.get('ratio_val')) == 0:

        train_assets, eval_assets, _, _, _ = dataset_version.train_test_split(prop=float(parameters.get('ratio_train')))
        try:
            train_assets.add_tags(dataset_version.create_asset_tag("train"))
        except:
            pass
        _,job_train = dataset_version.fork(version=dataset_version_name+"_train",assets=train_assets, type= dataset_version.type, with_tags = False ,with_labels = True, with_annotations = True, wait=False)
        job_train.wait_for_done(blocking_time_increment=5.0, attempts=360)
        try:
            eval_assets.add_tags(dataset_version.create_asset_tag("test"))
        except:
            pass
        _, job_test = dataset_version.fork(version=dataset_version_name+"_test",assets=eval_assets, type= dataset_version.type, with_tags = False ,with_labels = True, with_annotations = True, wait=False)
        job_test.wait_for_done(blocking_time_increment=5.0, attempts=360)

    elif float(parameters.get('ratio_test')) == 0:

        train_assets, eval_assets, _, _, _ = dataset_version.train_test_split(prop=float(parameters.get('ratio_train')))
        try:
            train_assets.add_tags(dataset_version.create_asset_tag("train"))
        except:
            pass
        _, job_train = dataset_version.fork(version=dataset_version_name+"_train",assets=train_assets, type= dataset_version.type, with_tags = False ,with_labels = True, with_annotations = True, wait=False)
        job_train.wait_for_done(blocking_time_increment=5.0, attempts=360)
        try:
            eval_assets.add_tags(dataset_version.create_asset_tag("val"))
        except:
            pass
        _,job_val = dataset_version.fork(version=dataset_version_name+"_val",assets=eval_assets, type= dataset_version.type, with_tags = False ,with_labels = True, with_annotations = True, wait=False)
        job_val.wait_for_done(blocking_time_increment=5.0, attempts=360)

        print(f"End of splitting")

def process(picsellia_model: Model, picsellia_dataset: CocoDataset):
    """
    🚀 This function processes the dataset using `process_images()`.

    🔹 **What You Need to Do:**
    - Modify `process_images()` to apply custom transformations or augmentations.
    - Ensure it returns the correct processed images & COCO metadata.

    Args:
        picsellia_model (Model): The model used for processing the dataset.
        picsellia_dataset (CocoDataset): The input dataset to be processed.

    Returns:
        CocoDataset: The processed dataset, ready for local execution and Picsellia.
    """

    # Get processing parameters from the user-defined configuration
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    parameters = context.processing_parameters.to_dict()

    if picsellia_dataset.dataset_version.type == InferenceType.NOT_CONFIGURED:
        picsellia_dataset.dataset_version.set_type(picsellia_model.model_version.type)
        picsellia_dataset.download_annotations(destination_dir=picsellia_dataset.annotations_dir, use_id=True)

    if picsellia_dataset.dataset_version.type != picsellia_model.model_version.type:
        raise ValueError(
            f"❌ Dataset type '{picsellia_dataset.dataset_version.type}' "
            f"does not match model type '{picsellia_model.model_version.type}'"
        )

    # Call the helper function to process images
    output_coco = process_images(
        picsellia_model=picsellia_model,
        picsellia_dataset=picsellia_dataset,
        parameters=parameters,
    )

    # Assign processed data to output dataset
    picsellia_dataset.coco_data = output_coco

    with open(picsellia_dataset.coco_file_path, "w") as f:
        json.dump(picsellia_dataset.coco_data, f)

    print("✅ Dataset processing complete!")
    return picsellia_dataset
