from copy import deepcopy

from picsellia_cv_engine.core import CocoDataset
from picsellia_cv_engine.core.contexts import PicselliaProcessingContext
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from utils.processing import process_images


@step
def process(input_dataset: CocoDataset, output_dataset: CocoDataset):
    """
    🚀 This function processes the dataset using `process_images()`.

    🔹 **What You Need to Do:**
    - Modify `process_images()` to apply custom transformations or augmentations.
    - Ensure it returns the correct processed images & COCO metadata.

    Args:
        input_dataset (CocoDataset): Input dataset from Picsellia.
        output_dataset (CocoDataset): Output dataset for saving processed data.

    Returns:
        CocoDataset: The processed dataset, ready for local execution and Picsellia.
    """

    # Get processing parameters from the user-defined configuration
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    parameters = context.processing_parameters.to_dict()

    # Initialize an empty COCO dataset
    output_coco = deepcopy(input_dataset.coco_data)
    output_coco["images"] = []  # Reset image metadata
    output_coco["annotations"] = []  # Reset annotation metadata

    # Call the helper function to process images
    output_coco = process_images(
        input_images_dir=input_dataset.images_dir,
        input_coco=input_dataset.coco_data,
        parameters=parameters,
        output_images_dir=output_dataset.images_dir,
        output_coco=output_coco,
        inference_type=input_dataset.dataset_version.type,
    )
    # Assign processed data to output dataset
    output_dataset.coco_data = output_coco

    print("✅ Dataset processing complete!")
    return output_dataset
