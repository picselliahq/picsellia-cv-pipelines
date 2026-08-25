import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from picsellia.types.enums import InferenceType
from picsellia_cv_engine.core import CocoDataset
from picsellia_cv_engine.core.contexts import PicselliaDatasetProcessingContext
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from transformers import Sam3Model, Sam3Processor
from utils.sam3_processing import process_images_sam3


@step
def load_sam3_model():
    """
    Load SAM-3 model and processor from Hugging Face.

    Returns:
        tuple: (Sam3Model, Sam3Processor) ready for inference
    """
    import sys
    import tempfile

    # Set HuggingFace cache directory FIRST, before any HF operations
    # This prevents permission errors in Docker environments
    hf_cache_dir = os.environ.get("HF_HOME")
    if not hf_cache_dir:
        # Use /tmp or system temp directory if HF_HOME not set
        hf_cache_dir = os.path.join(tempfile.gettempdir(), "huggingface")
        os.environ["HF_HOME"] = hf_cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
        os.makedirs(hf_cache_dir, exist_ok=True)
        print(f"📁 Set HuggingFace cache directory to: {hf_cache_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load environment variables from .env file
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        print(f"ℹ️ No .env file found at {env_path}")

    # Monkey-patch sys.stdout to add isatty() method if missing
    # This fixes compatibility with StreamToLogger used by the pipeline
    if not hasattr(sys.stdout, "isatty"):
        sys.stdout.isatty = lambda: False
    if not hasattr(sys.stderr, "isatty"):
        sys.stderr.isatty = lambda: False

    # Handle HuggingFace authentication
    # Check if token is available in environment or already logged in
    try:
        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get(
            "HF_TOKEN"
        )
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)
            print("✅ Logged in to Hugging Face using token from environment")
        else:
            print(
                "ℹ️ No HF_TOKEN found in environment, attempting to use cached credentials"
            )
    except Exception as e:
        print(f"⚠️ Could not login to Hugging Face: {e}")
        print("   Attempting to load model without explicit login...")

    print(f"🔄 Loading SAM-3 model on device: {device}")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    print("✅ SAM-3 model loaded successfully!")

    return (model, processor)


@step
def process(
    model: Sam3Model,
    processor: Sam3Processor,
    picsellia_dataset: CocoDataset,
):
    """
    Process the dataset using SAM-3 segmentation model.

    This function:
    - Takes a SAM-3 model and processor
    - Processes images from the Picsellia dataset
    - Generates segmentation masks and bounding boxes
    - Saves results in COCO format

    Args:
        model (Sam3Model): The SAM-3 model for segmentation.
        processor (Sam3Processor): The SAM-3 processor for input preparation.
        picsellia_dataset (CocoDataset): The input dataset to be processed.

    Returns:
        CocoDataset: The processed dataset with segmentation annotations.
    """

    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    parameters = context.processing_parameters
    text_prompt = context.inputs.get("text_prompt")
    label_name = context.inputs.get("label_name")
    box_prompt = parameters.to_dict().get("box_prompt")

    if text_prompt is None and box_prompt is None:
        raise ValueError(
            "❌ At least one of 'text_prompt' or 'box_prompt' must be provided in processing parameters.\n"
            "Example parameters:\n"
            "  - text_prompt: 'waste'\n"
            "  - box_prompt: [100, 100, 500, 500]\n"
            "  - threshold: 0.5\n"
            "  - mask_threshold: 0.5\n"
            "  - label_name: 'object'"
        )

    # Validate dataset type
    dataset_type = picsellia_dataset.dataset_version.type
    if dataset_type not in [InferenceType.NOT_CONFIGURED, InferenceType.SEGMENTATION]:
        raise ValueError(
            f"❌ Invalid dataset type: {dataset_type}\n"
            f"   SAM-3 segmentation pipeline only supports datasets with type:\n"
            f"   - NOT_CONFIGURED (will be set to SEGMENTATION)\n"
            f"   - SEGMENTATION\n"
            f"   Current dataset type: {dataset_type}"
        )

    # Set dataset type if not configured
    if picsellia_dataset.dataset_version.type == InferenceType.NOT_CONFIGURED:
        # SAM-3 produces segmentation, so set to SEGMENTATION type
        picsellia_dataset.dataset_version.set_type(InferenceType.SEGMENTATION)
        picsellia_dataset.download_annotations(
            destination_dir=picsellia_dataset.annotations_dir, use_id=True
        )

    params = parameters.to_dict()
    params["text_prompt"] = text_prompt
    params["label_name"] = label_name

    print("PARAMS:", params)
    # Call the helper function to process images with SAM-3
    output_coco = process_images_sam3(
        sam3_model=model,
        sam3_processor=processor,
        picsellia_dataset=picsellia_dataset,
        parameters=params,
    )

    # Assign processed data to output dataset
    picsellia_dataset.coco_data = output_coco

    # Save COCO annotations to file
    with open(picsellia_dataset.coco_file_path, "w") as f:
        json.dump(picsellia_dataset.coco_data, f, indent=2)

    print("✅ Dataset processing complete!")
    print(f"   💾 COCO file saved to: {picsellia_dataset.coco_file_path}")

    return picsellia_dataset
