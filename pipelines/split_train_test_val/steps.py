from datetime import datetime

from picsellia.exceptions import PicselliaError, ResourceConflictError
from picsellia.types.enums import InferenceType
from picsellia_cv_engine.core.contexts import PicselliaDatasetProcessingContext
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step

from utils.parameters import ProcessingParameters


@step
def length_dataset_version_sanity_check() -> bool:
    context: PicselliaDatasetProcessingContext[ProcessingParameters] = Pipeline.get_active_context()
    print("Check of the DatasetVersion length, it must contain at least 3 Assets.")
    assert len(context.input_dataset_version.list_assets()) >= 3, PicselliaError(
        "The DatasetVersion has less than 3 assets."
    )
    return True


@step
def parameters_sanity_chek() -> bool:
    context: PicselliaDatasetProcessingContext[ProcessingParameters] = Pipeline.get_active_context()
    ratio_train = float(context.inputs.get("ratio_train"))
    ratio_test = float(context.inputs.get("ratio_test"))
    ratio_val = float(context.inputs.get("ratio_val"))
    print(
        "Check of the parameters, ratios should sum to 1 and train_ratio must be greater than 0"
    )
    assert (
        ratio_train + ratio_val + ratio_test == 1,
        PicselliaError(
            f"The sum of the three ratios is not 1 but {ratio_train + ratio_val + ratio_test}"
        ),
    )
    assert (
        ratio_train > 0,
        PicselliaError("The train dataset cannot be empty"),
    )
    assert (
        ratio_test >= 0,
        PicselliaError("The parameter test_ratio must be greater than 0."),
    )
    assert (
        ratio_val >= 0,
        PicselliaError("The parameter val_ratio must be greater than 0."),
    )
    assert (
        ratio_val + ratio_test >= 0,
        PicselliaError("Either ratio_val or ratio_test must be strictly greater than 0"),
    )
    return True


@step
def create_empty_annotation() -> bool:
    context: PicselliaDatasetProcessingContext[ProcessingParameters] = Pipeline.get_active_context()
    parameters = context.processing_parameters

    dataset_version = context.input_dataset_version
    if parameters.embed_asset_without_annotation:
        print("Creating empty annotations for assets without existing annotations")
        assert dataset_version != InferenceType.NOT_CONFIGURED, PicselliaError(
            "The DatasetVersion type should be configured to create empty annotations"
        )
        empty_assets = [
            asset
            for asset in dataset_version.list_assets()
            if len(asset.list_annotations()) == 0
        ]
        for empty_asset in empty_assets:
            empty_asset.create_annotation()
        print(f"{str(len(empty_assets))} empty annotations created")


@step
def split_and_tag_data() -> bool:
    context: PicselliaDatasetProcessingContext[ProcessingParameters] = Pipeline.get_active_context()
    parameters = context.processing_parameters
    dataset_version = context.input_dataset_version
    dataset_version_name = dataset_version.version

    if parameters.add_asset_tags:
        print('Creation of the Asset Tags "train", "test" and "val" if not existing')

        try:
            train_tag = dataset_version.create_asset_tag("train")
        except Exception:
            train_tag = dataset_version.get_asset_tag("train")

        try:
            test_tag = dataset_version.create_asset_tag("test")
        except Exception:
            test_tag = dataset_version.get_asset_tag("test")

        try:
            val_tag = dataset_version.create_asset_tag("val")
        except Exception:
            val_tag = dataset_version.get_asset_tag("val")

    ratio_train = float(context.inputs.get("ratio_train"))
    ratio_test = float(context.inputs.get("ratio_test"))
    ratio_val = float(context.inputs.get("ratio_val"))

    print("Start splitting the DatasetVersion")

    if ratio_val != 0 and ratio_test != 0:
        train_assets, test_assets, val_assets, _, _, _, _ = dataset_version.train_test_val_split(
            ratios=[ratio_train, ratio_test, ratio_val]
        )
    elif ratio_val == 0 and ratio_test != 0:
        train_assets, test_assets, _, _, _ = dataset_version.train_test_split(prop=ratio_train)
        val_assets = []
    elif ratio_val != 0 and ratio_test == 0:
        train_assets, val_assets, _, _, _ = dataset_version.train_test_split(prop=ratio_train)
        test_assets = []

    if len(train_assets) != 0:
        if parameters.add_asset_tags:
            try:
                train_assets.add_tags(train_tag)
                print(f"Adding tag {train_tag.name} on {len(train_assets)} Assets from input DatasetVersion")
            except Exception:
                pass
        try:
            _, job_train = dataset_version.fork(
                version=dataset_version_name + "_train",
                assets=train_assets,
                type=dataset_version.type,
                with_tags=False,
                with_labels=True,
                with_annotations=True,
                wait=False,
            )
            job_train.wait_for_done(blocking_time_increment=5.0, attempts=360)
            print(f'DatasetVersion with name "{dataset_version_name}_train" created.')
        except ResourceConflictError:
            print(f'A DatasetVersion with name "{dataset_version_name}_train" already exists, adding a timestamp to ensure unicity')
            timestamped_name_train = dataset_version_name + "_train_" + str(datetime.now().timestamp())
            _, job_train = dataset_version.fork(
                version=timestamped_name_train,
                assets=train_assets,
                type=dataset_version.type,
                with_tags=False,
                with_labels=True,
                with_annotations=True,
                wait=False,
            )
            job_train.wait_for_done(blocking_time_increment=5.0, attempts=360)
            print(f'DatasetVersion with name "{timestamped_name_train}" created.')

    if len(test_assets) != 0:
        if parameters.add_asset_tags:
            try:
                test_assets.add_tags(test_tag)
                print(f"Adding tag {test_tag.name} on {len(test_assets)} Assets from input DatasetVersion")
            except Exception:
                pass
        try:
            _, job_test = dataset_version.fork(
                version=dataset_version_name + "_test",
                assets=test_assets,
                type=dataset_version.type,
                with_tags=False,
                with_labels=True,
                with_annotations=True,
                wait=False,
            )
            job_test.wait_for_done(blocking_time_increment=5.0, attempts=360)
            print(f'DatasetVersion with name "{dataset_version_name}_test" created.')
        except ResourceConflictError:
            print(f'A DatasetVersion with name "{dataset_version_name}_test" already exists, adding a timestamp to ensure unicity')
            timestamped_name_test = dataset_version_name + "_test_" + str(datetime.now().timestamp())
            _, job_test = dataset_version.fork(
                version=timestamped_name_test,
                assets=test_assets,
                type=dataset_version.type,
                with_tags=False,
                with_labels=True,
                with_annotations=True,
                wait=False,
            )
            job_test.wait_for_done(blocking_time_increment=5.0, attempts=360)
            print(f'DatasetVersion with name "{timestamped_name_test}" created.')

    if len(val_assets) != 0:
        if parameters.add_asset_tags:
            try:
                val_assets.add_tags(val_tag)
                print(f"Adding tag {val_tag.name} on {len(val_assets)} Assets from input DatasetVersion")
            except Exception:
                pass
        try:
            _, job_val = dataset_version.fork(
                version=dataset_version_name + "_val",
                assets=val_assets,
                type=dataset_version.type,
                with_tags=False,
                with_labels=True,
                with_annotations=True,
                wait=False,
            )
            job_val.wait_for_done(blocking_time_increment=5.0, attempts=360)
            print(f'DatasetVersion with name "{dataset_version_name}_val" created.')
        except ResourceConflictError:
            print(f'A DatasetVersion with name "{dataset_version_name}_val" already exists, adding a timestamp to ensure unicity')
            timestamped_name_val = dataset_version_name + "_val_" + str(datetime.now().timestamp())
            _, job_val = dataset_version.fork(
                version=timestamped_name_val,
                assets=val_assets,
                type=dataset_version.type,
                with_tags=False,
                with_labels=True,
                with_annotations=True,
                wait=False,
            )
            job_val.wait_for_done(blocking_time_increment=5.0, attempts=360)
            print(f'DatasetVersion with name "{timestamped_name_val}" created.')

    print("End of splitting")
