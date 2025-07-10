import dataclasses
from argparse import ArgumentParser

from picsellia_cv_engine.core.contexts.processing.datalake.local_datalake_processing_context import (
    LocalDatalakeProcessingContext,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.datalake.loader import load_datalake

from pipelines.datalake_autotagging.pipeline_utils.steps.model_loading.clip_model_context_loader import (
    load_clip_model,
)
from pipelines.datalake_autotagging.pipeline_utils.steps.processing.clip_datalake_autotagging import (
    autotag_datalake_with_clip,
)
from pipelines.datalake_autotagging.pipeline_utils.steps.weights_extraction.hugging_face_weights_extractor import (
    get_hugging_face_model,
)


@dataclasses.dataclass
class ProcessingDatalakeAutotaggingParameters:
    tags_list: list[str]
    device: str
    batch_size: int


parser = ArgumentParser()
parser.add_argument("--api_token", type=str)
parser.add_argument("--organization_id", type=str)
parser.add_argument("--job_id", type=str)
parser.add_argument("--input_datalake_id", type=str)
parser.add_argument("--output_datalake_id", type=str, required=False)
parser.add_argument("--model_version_id", type=str)
parser.add_argument("--tags_list", nargs="+", type=str)
parser.add_argument("--offset", type=int, default=0)
parser.add_argument("--limit", type=int, default=100)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()


def get_context() -> LocalDatalakeProcessingContext:
    return LocalDatalakeProcessingContext(
        api_token=args.api_token,
        organization_id=args.organization_id,
        job_id=args.job_id,
        job_type=None,
        input_datalake_id=args.input_datalake_id,
        output_datalake_id=args.output_datalake_id,
        model_version_id=args.model_version_id,
        offset=args.offset,
        limit=args.limit,
        use_id=True,
        processing_parameters=ProcessingDatalakeAutotaggingParameters(
            tags_list=args.tags_list, device=args.device, batch_size=args.batch_size
        ),
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def datalake_autotagging_processing_pipeline() -> None:
    datalake = load_datalake()
    model = get_hugging_face_model()
    model = load_clip_model(model=model, device="cuda:0")
    autotag_datalake_with_clip(datalake=datalake, model=model, device="cuda:0")


if __name__ == "__main__":
    import os

    import torch

    cpu_count = os.cpu_count()
    if cpu_count is not None and cpu_count > 1:
        torch.set_num_threads(cpu_count - 1)

    datalake_autotagging_processing_pipeline()
