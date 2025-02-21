import json
import os

from picsellia.services.error_manager import ErrorManager
from picsellia.types.enums import AnnotationFileType
from pycocotools.coco import COCO


def load_input_output_annotations(input_dataset_version):
    input_annotations_path = input_dataset_version.export_annotation_file(
        annotation_file_type=AnnotationFileType.COCO,
        target_path=os.path.join("annotations"),
        force_replace=True,
    )

    input_annotations_file = open(input_annotations_path)

    input_annotations_coco = COCO(input_annotations_path)

    input_annotations_dict = json.load(input_annotations_file)
    output_annotations_dict = input_annotations_dict.copy()

    output_annotations_dict["images"] = []
    output_annotations_dict["annotations"] = []

    return input_annotations_dict, input_annotations_coco, output_annotations_dict


def upload_data_to_datalake(client, filepaths):
    datalake = client.get_datalake()

    error_manager = ErrorManager()
    data_list = datalake.upload_data(
        filepaths, tags=["augmented", "processing"], error_manager=error_manager
    )
    error_paths = [error.path for error in error_manager.errors]

    while len(error_paths) != 0:
        error_manager.clear()
        data_list.append(
            datalake.upload_data(
                error_paths,
                tags=["augmented", "processing"],
                error_manager=error_manager,
            )
        )
        error_paths = [error.path for error in error_manager.errors]

    return data_list


def write_annotations_dict(annotations_dict, annotations_path):
    with open(annotations_path, "w") as outfile:
        json.dump(annotations_dict, outfile)


def list_filepaths_from_folder(folder_path):
    filenames = os.listdir(folder_path)
    filepaths = [os.path.join(folder_path, filename) for filename in filenames]
    return filepaths
