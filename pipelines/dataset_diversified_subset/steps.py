import numpy as np
from picsellia.sdk.dataset_version import DatasetVersion
from picsellia_cv_engine.core.contexts import PicselliaDatasetProcessingContext
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from utils.inputs import DiversitySelectionAlgorithm
from utils.processing import (
    farthest_point_sampling,
    fetch_embeddings,
    fork_dataset_subset,
    kmeans_diverse_subset,
)


@step
def fetch_dataset_embeddings() -> tuple[list[str], np.ndarray]:
    """
    Pull the embeddings computed by the platform (Visual Search) for every
    indexed asset of the target dataset version.
    """
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    parameters = context.processing_parameters
    dataset_version = context.target

    data_ids, vectors = fetch_embeddings(dataset_version, parameters.embedder_key)
    print(f"Fetched vectors: {vectors.shape[0]} x dim={vectors.shape[1]}")
    return data_ids, vectors


@step
def select_diverse_subset(data_ids: list[str], vectors: np.ndarray) -> list[str]:
    """
    Select a diverse subset of assets from the fetched embeddings, using the
    algorithm requested through the 'algorithm' input ('fps' or 'kmeans').
    """
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    parameters = context.processing_parameters

    n_samples = int(context.inputs.get("n_samples"))
    if n_samples > len(data_ids):
        raise ValueError(
            f"n_samples ({n_samples}) > number of assets with embeddings ({len(data_ids)})"
        )

    algorithm = DiversitySelectionAlgorithm(context.inputs.get("algorithm").lower())
    print(f"Selecting {n_samples} diverse images (algorithm: {algorithm.value})...")

    if algorithm == DiversitySelectionAlgorithm.FPS:
        selected_indices = farthest_point_sampling(
            vectors, n_samples, seed=parameters.seed
        )
    else:
        selected_indices = kmeans_diverse_subset(
            vectors, n_samples, seed=parameters.seed
        )
        if len(selected_indices) < n_samples:
            print(
                f"Warning: {len(selected_indices)} non-empty clusters out of "
                f"{n_samples} requested (some clusters were empty)."
            )

    return [data_ids[i] for i in selected_indices]



@step
def create_subset_dataset_version(selected_data_ids: list[str]) -> DatasetVersion:
    """
    Fork the target dataset version, keeping only the selected assets, into a
    new dataset version named after the 'target_version_name' input.
    """
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    parameters = context.processing_parameters
    dataset_version = context.target

    target_version_name = context.inputs.get("target_version_name")

    return fork_dataset_subset(
        dataset_version=dataset_version,
        selected_data_ids=selected_data_ids,
        new_version_name=target_version_name,
        with_annotations=parameters.with_annotations,
        with_tags=parameters.with_tags,
    )
