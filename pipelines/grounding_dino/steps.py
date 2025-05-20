import json
from copy import deepcopy

from picsellia_cv_engine.core import CocoDataset, Model
from picsellia_cv_engine.core.contexts import PicselliaProcessingContext
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step

from pipelines.grounding_dino.utils.processing import process_images


@step
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
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    parameters = context.processing_parameters.to_dict()

    # Initialize an empty COCO dataset
    output_coco = deepcopy(picsellia_dataset.coco_data)
    output_coco["images"] = []  # Reset image metadata
    output_coco["annotations"] = []  # Reset annotation metadata

    # Call the helper function to process images
    output_coco = process_images(
        picsellia_model=picsellia_model,
        images_dir=picsellia_dataset.images_dir,
        coco=picsellia_dataset.coco_data,
        parameters=parameters,
    )
    print(f"output_coco: {output_coco}")
    # Assign processed data to output dataset
    picsellia_dataset.coco_data = output_coco

    with open(picsellia_dataset.coco_file_path, "w") as f:
        json.dump(picsellia_dataset.coco_data, f)

    print("✅ Dataset processing complete!")
    return picsellia_dataset
