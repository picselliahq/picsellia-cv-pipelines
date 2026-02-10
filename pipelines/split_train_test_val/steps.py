from picsellia.types.enums import InferenceType
from picsellia_cv_engine.core.contexts import PicselliaDatasetProcessingContext
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia.exceptions import ResourceConflictError 

from datetime import datetime

from picsellia.exceptions import PicselliaError

@step
def length_dataset_version_sanity_check()-> bool:
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    print(f"Verification of the DatasetVersion length, DatasetVersion must contains at 3 Assets.")
    assert len(context.input_dataset_version.list_assets()) >= 3, PicselliaError(f"The DatasetVersion has less than 3 assets.")
    return True

@step
def parameters_sanity_chek()-> bool:
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    parameters = context.processing_parameters.to_dict()
    print(f"Verification of the parameters, the sum of the ratios should be equal to 1 and train_ratio cannot bo 0")
    assert (parameters.get('ratio_train')+parameters.get('ratio_val')+parameters.get('ratio_test') == 1, PicselliaError(f"The sum of the three ratios is not 1 but {parameters.get('ratio_train')+parameters.get('ratio_val')+parameters.get('ratio_test')}"))
    assert (parameters.get('ratio_train') != 0, PicselliaError(f"The train dataset cannot be empty"))
    return True


@step
def create_empty_annotation()->bool:
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    parameters = context.processing_parameters.to_dict()

    dataset_version = context.input_dataset_version
    if parameters.get('embed_asset_without_annotation')== True:
        print(f"Creating empty annotations for assets without existing annotations")
        assert dataset_version!=InferenceType.NOT_CONFIGURED, PicselliaError(f"The DatasetVersion type should be configured to create empty annotations")
        empty_assets = [asset for asset in dataset_version.list_assets() if len(asset.list_annotations())==0]
        for empty_asset in empty_assets:
            empty_asset.create_annotation()
        print(f"{str(len(empty_assets))} empty annotations created")


@step   
def split_and_tag_data()-> bool:
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    parameters = context.processing_parameters.to_dict()
    dataset_version = context.input_dataset_version
    dataset_version_name = dataset_version.version

    if parameters.get("add_asset_tags")==True:
        
        print(f"Creation of the Asset Tags \"train\", \"test\" and \"val\" if not existing")

        try:
            train_tag = dataset_version.create_asset_tag("train")
        except:
            train_tag = dataset_version.get_asset_tag("train")

        try:
            test_tag = dataset_version.create_asset_tag("test")
        except:
            test_tag = dataset_version.get_asset_tag("test")
        try:
            val_tag = dataset_version.create_asset_tag("val")
        except:
            val_tag = dataset_version.get_asset_tag("val")

    
    print(f"Start splitting the DatasetVersion")
    
    if float(parameters.get('ratio_val')) != 0 and float(parameters.get('ratio_test')) != 0:
        train_assets, test_assets, val_assets, _, _, _, _ = dataset_version.train_test_val_split(ratios=[parameters.get('ratio_train'),parameters.get('ratio_test'),parameters.get('ratio_val')])
        
        if len(train_assets)!=0:
            if parameters.get("add_asset_tags")== True:
                try:
                    train_assets.add_tags(train_tag)
                    print(f"Adding tag {train_tag.name} on {len(train_assets)} Assets from input DatasetVersion")
                except:
                    pass
            try:
                _,job_train = dataset_version.fork(version=dataset_version_name+"_train",assets=train_assets, type= dataset_version.type, with_tags = False ,with_labels = True, with_annotations = True, wait=False)
                job_train.wait_for_done(blocking_time_increment=5.0, attempts=360)
            except ResourceConflictError:
                print(f"A DatasetVersion with name \"{dataset_version_name}_train\" already exists, adding a timestamp to the name of the created DatasetVersion name to ensure unicity")
                _,job_train = dataset_version.fork(version=dataset_version_name+"_train_"+str(datetime.now().timestamp()),assets=train_assets, type= dataset_version.type, with_tags = False ,with_labels = True, with_annotations = True, wait=False)
                job_train.wait_for_done(blocking_time_increment=5.0, attempts=360)


        if len(test_assets)!=0:
            if parameters.get("add_asset_tags")== True:
                try:
                    test_assets.add_tags(test_tag)
                    print(f"Adding tag {test_tag.name} on {len(test_assets)} Assets from input DatasetVersion")
                except:
                    pass
            try:
                _,job_test = dataset_version.fork(version=dataset_version_name+"_test",assets=test_assets, type= dataset_version.type, with_tags = False ,with_labels = True, with_annotations = True, wait=False)
                job_test.wait_for_done(blocking_time_increment=5.0, attempts=360)
            except ResourceConflictError:
                print(f"A DatasetVersion with name \"{dataset_version_name}_test\" already exists, adding a timestamp to the name of the created DatasetVersion name to ensure unicity")
                _,job_test = dataset_version.fork(version=dataset_version_name+"_test_"+str(datetime.now().timestamp()),assets=test_assets, type= dataset_version.type, with_tags = False ,with_labels = True, with_annotations = True, wait=False)
                job_test.wait_for_done(blocking_time_increment=5.0, attempts=360)


        if len(val_assets)!=0:
            if parameters.get("add_asset_tags")== True:
                try:
                    val_assets.add_tags(val_tag)
                    print(f"Adding tag {val_tag.name} on {len(val_assets)} Assets from input DatasetVersion")
                except:
                    pass
            try:
                _,job_val = dataset_version.fork(version=dataset_version_name+"_val",assets=val_assets, type= dataset_version.type, with_tags = False ,with_labels = True, with_annotations = True, wait=False)
                job_val.wait_for_done(blocking_time_increment=5.0, attempts=360)
            except ResourceConflictError:
                print(f"A DatasetVersion with name \"{dataset_version_name}_val\" already exists, adding a timestamp to the name of the created DatasetVersion name to ensure unicity")
                _,job_val = dataset_version.fork(version=dataset_version_name+"_val_"+str(datetime.now().timestamp()),assets=val_assets, type= dataset_version.type, with_tags = False ,with_labels = True, with_annotations = True, wait=False)
                job_val.wait_for_done(blocking_time_increment=5.0, attempts=360)

        print(f"End of splitting")
