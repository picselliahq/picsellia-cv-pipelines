import json
import os
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from picsellia import DatasetVersion
from picsellia.exceptions import ResourceConflictError
from picsellia.types.enums import ImportAnnotationMode, InferenceType
from picsellia_cv_engine.core import CocoDataset
from picsellia_cv_engine.core.contexts import PicselliaDatasetProcessingContext
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from transformers import Sam3Model, Sam3Processor
from utils.sam3_processing import process_boxes_to_polygons

ANNOTATION_MODES = {
    "keep": ImportAnnotationMode.KEEP,
    "replace": ImportAnnotationMode.REPLACE,
    "concatenate": ImportAnnotationMode.CONCATENATE,
}


@step
def fork_dataset() -> DatasetVersion:
    """
    Fork the input (bbox-annotated) dataset version into a new SEGMENTATION
    dataset version. Forking attaches the same underlying images server-side,
    so no image file is downloaded or re-uploaded - only the annotations
    change (boxes -> polygons).
    """
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    input_dataset_version = context.input_dataset_version

    if input_dataset_version.type not in (
        InferenceType.OBJECT_DETECTION,
        InferenceType.NOT_CONFIGURED,
    ):
        raise ValueError(
            "❌ Input dataset version must be annotated with bounding boxes "
            f"(type OBJECT_DETECTION or NOT_CONFIGURED), got {input_dataset_version.type}."
        )

    version_name = context.inputs.get("output_dataset_version_name")
    if not version_name:
        raise ValueError("Input 'output_dataset_version_name' is required.")

    try:
        output_dataset_version, job = input_dataset_version.fork(
            version=version_name,
            type=InferenceType.SEGMENTATION,
            with_tags=True,
            with_labels=False,
            with_annotations=False,
            wait=False,
        )
    except ResourceConflictError:
        version_name = f"{version_name}_{datetime.now().timestamp()}"
        print(
            f"A dataset version with that name already exists, forking as "
            f"'{version_name}' instead."
        )
        output_dataset_version, job = input_dataset_version.fork(
            version=version_name,
            type=InferenceType.SEGMENTATION,
            with_tags=True,
            with_labels=False,
            with_annotations=False,
            wait=False,
        )
    job.wait_for_done(attempts=1000)

    for label in input_dataset_version.list_labels():
        output_dataset_version.create_label(label.name)

    print(
        f"Forked '{input_dataset_version.version}' into '{version_name}' "
        f"(type SEGMENTATION) without re-uploading data, and recreated its labels."
    )
    return output_dataset_version


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
    output_dataset_version: DatasetVersion,
) -> str:
    """
    Download the input dataset's images and bounding box annotations, run
    SAM-3 with each existing box as a prompt to generate a polygon mask
    (keeping the box's original label), and save the result as a COCO file.

    Returns:
        str: Path to the generated COCO annotation file.
    """
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    input_dataset_version = context.input_dataset_version
    parameters = context.processing_parameters.to_dict()

    input_dataset = CocoDataset(name="input", dataset_version=input_dataset_version)
    input_dataset.download_annotations(
        destination_dir=os.path.join(context.working_dir, "annotations", "input"),
        use_id=True,
    )
    input_dataset.download_assets(
        destination_dir=os.path.join(context.working_dir, "images", "input"),
        use_id=True,
    )

    output_coco = process_boxes_to_polygons(
        sam3_model=model,
        sam3_processor=processor,
        picsellia_dataset=input_dataset,
        parameters=parameters,
    )

    output_annotations_dir = os.path.join(
        context.working_dir, "annotations", "output"
    )
    os.makedirs(output_annotations_dir, exist_ok=True)
    coco_file_path = os.path.join(output_annotations_dir, "annotations.json")
    with open(coco_file_path, "w") as f:
        json.dump(output_coco, f, indent=2)

    print("✅ Dataset processing complete!")
    print(f"   💾 COCO file saved to: {coco_file_path}")

    return coco_file_path


@step
def upload_annotations(coco_file_path: str, output_dataset_version: DatasetVersion):
    """
    Upload the generated polygon annotations onto the output dataset version,
    honoring the configured annotation_mode (keep / replace / concatenate).
    """
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    annotation_mode = context.processing_parameters.annotation_mode.lower()

    if annotation_mode not in ANNOTATION_MODES:
        raise ValueError(
            f"❌ Invalid annotation_mode '{annotation_mode}'. "
            f"Must be one of: {list(ANNOTATION_MODES)}"
        )

    output_dataset_version.import_annotations_coco_file(
        file_path=coco_file_path,
        use_id=True,
        fail_on_asset_not_found=True,
        mode=ANNOTATION_MODES[annotation_mode],
    )
