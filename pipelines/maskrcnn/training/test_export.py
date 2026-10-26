"""Test script to download checkpoint from a Picsellia experiment and export it.

Usage:
    python test_export.py \
        --api-token YOUR_TOKEN \
        --organization-name YOUR_ORG \
        --project-name YOUR_PROJECT \
        --experiment-name YOUR_EXPERIMENT \
        --export-format all \
        --backbone resnet50

    Environment variables can be used instead of CLI args:
        PICSELLIA_API_TOKEN, PICSELLIA_ORGANIZATION_NAME
"""

import argparse
import os
import tempfile

import torch
from picsellia import Client
from utils.steps_utils import (
    export_to_onnx,
    export_to_torchscript,
    get_maskrcnn_model,
)


def main():
    parser = argparse.ArgumentParser(description="Export MaskRCNN from Picsellia experiment")
    parser.add_argument("--api-token", type=str, default=os.getenv("PICSELLIA_API_TOKEN"))
    parser.add_argument("--organization-name", type=str, default=os.getenv("PICSELLIA_ORGANIZATION_NAME"))
    parser.add_argument("--host", type=str, default=os.getenv("PICSELLIA_HOST", "https://app.picsellia.com"))
    parser.add_argument("--project-name", type=str, required=True)
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--export-format", type=str, default="all", choices=["torchscript", "onnx", "all"])
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--output-dir", type=str, default="./export_output")
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a real image for ONNX tracing (ensures mask branch is fully exercised)",
    )
    args = parser.parse_args()

    if not args.api_token:
        raise ValueError("API token required (--api-token or PICSELLIA_API_TOKEN env var)")

    print("=" * 60)
    print("MASKRCNN EXPORT TEST")
    print("=" * 60)

    print("\n[1/4] Connecting to Picsellia...")
    client = Client(
        api_token=args.api_token,
        organization_name=args.organization_name,
        host=args.host,
    )
    project = client.get_project(project_name=args.project_name)
    experiment = project.get_experiment(name=args.experiment_name)
    print(f"  - Project: {args.project_name}")
    print(f"  - Experiment: {args.experiment_name}")

    print("\n[2/4] Downloading checkpoint...")
    artifacts = experiment.list_artifacts()
    checkpoint_artifact = None
    for artifact in artifacts:
        if artifact.name == "checkpoint-latest":
            checkpoint_artifact = artifact
            break

    if checkpoint_artifact is None:
        raise RuntimeError("No 'checkpoint-latest' artifact found in experiment")

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_artifact.download(tmp_dir)
        checkpoint_path = os.path.join(tmp_dir, checkpoint_artifact.filename)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        num_classes = checkpoint["num_classes"]
        id2label = checkpoint["id2label"]
        backbone = checkpoint.get("backbone", args.backbone)
        image_size = checkpoint.get("image_size", 800)

        print(f"  - Classes: {num_classes}")
        print(f"  - Labels: {list(id2label.values())}")
        print(f"  - Backbone: {backbone}")
        print(f"  - Image size: {image_size}")

        print("\n[3/5] Loading model...")
        model = get_maskrcnn_model(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"  - Device: {device}")

        # Resolve trace image: use --image if provided, otherwise download
        # one from the experiment's training dataset.
        trace_image_path = args.image
        trace_image_dir = None
        if trace_image_path is None and args.export_format in ("onnx", "all"):
            print("\n[4/5] Downloading a training image for ONNX tracing...")
            attached = experiment.list_attached_dataset_versions()
            if attached:
                ds = attached[0]
                assets = ds.list_assets(limit=1)
                if assets:
                    trace_image_dir = tempfile.mkdtemp(dir=tmp_dir)
                    assets[0].download(trace_image_dir)
                    for fname in os.listdir(trace_image_dir):
                        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                            trace_image_path = os.path.join(trace_image_dir, fname)
                            break
            if trace_image_path:
                print(f"  - Trace image: {trace_image_path}")
            else:
                print("  - WARNING: No training image found, using random noise for tracing")

        print("\n[5/5] Exporting...")
        os.makedirs(args.output_dir, exist_ok=True)

        if args.export_format in ("torchscript", "all"):
            ts_path = os.path.join(args.output_dir, "model.torchscript")
            export_to_torchscript(model, ts_path, device, image_size)

        if args.export_format in ("onnx", "all"):
            onnx_path = os.path.join(args.output_dir, "model.onnx")
            export_to_onnx(
                model, onnx_path, device, image_size,
                trace_image_path=trace_image_path,
            )

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
