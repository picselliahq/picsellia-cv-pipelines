import numpy as np
from picsellia import Client
from picsellia.exceptions import ResourceConflictError, ResourceNotFoundError
from picsellia.sdk.dataset_version import DatasetVersion
from sklearn.cluster import KMeans


def fetch_embeddings(
    dataset_version: DatasetVersion, embedder_key: str | None
) -> tuple[list[str], np.ndarray]:
    """Fetch embeddings of all indexed assets of a DatasetVersion.

    Note: `list_embeddings` returns the UUID of the underlying (Data), not
    the UUID of the (Asset). Assets are resolved afterwards via
    `dataset_version.list_assets(data_ids=...)`.

    Returns:
        (data_ids, vectors) where vectors has shape (n_assets, dim)
    """
    count = dataset_version.count_embeddings()
    if count == 0:
        raise RuntimeError(
            "No embeddings found for this DatasetVersion. "
            "Activate Visual Search first with dataset_version.activate_visual_search()."
        )

    print(f"{count} indexed embeddings found, fetching...")
    points = dataset_version.list_embeddings(limit=count)

    available_keys = sorted({key for p in points for key in p["vector"]})
    if embedder_key is None:
        if len(available_keys) > 1:
            raise RuntimeError(
                "Several embedding models are available on this DatasetVersion: "
                f"{available_keys}. Specify which one to use with the "
                "'embedder_key' parameter."
            )
        embedder_key = available_keys[0]
    elif embedder_key not in available_keys:
        raise RuntimeError(
            f"embedder_key '{embedder_key}' not found. Available keys: {available_keys}"
        )

    data_ids = []
    vectors = []
    for p in points:
        if embedder_key not in p["vector"]:
            continue
        data_ids.append(p["id"])
        vectors.append(p["vector"][embedder_key])

    if not data_ids:
        raise RuntimeError(f"No asset has a vector for key '{embedder_key}'.")

    return data_ids, np.asarray(vectors, dtype=np.float32)


def farthest_point_sampling(
    vectors: np.ndarray, n_samples: int, seed: int = 0
) -> list[int]:
    """Select n_samples indices maximizing the minimum distance to the
    already-chosen subset (core-set selection).

    Complexity: O(n_samples * n_vectors), vectorized with numpy.
    """
    n = len(vectors)
    if n_samples >= n:
        return list(range(n))

    rng = np.random.default_rng(seed)
    start = int(rng.integers(n))
    selected = [start]

    min_dist = np.linalg.norm(vectors - vectors[start], axis=1)
    for _ in range(1, n_samples):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        new_dist = np.linalg.norm(vectors - vectors[next_idx], axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    return selected


def kmeans_diverse_subset(
    vectors: np.ndarray, n_samples: int, seed: int = 0
) -> list[int]:
    """Cluster embeddings into n_samples clusters and return, for each
    non-empty cluster, the index of the vector closest to its centroid.
    """
    n = len(vectors)
    if n_samples >= n:
        return list(range(n))

    kmeans = KMeans(n_clusters=n_samples, random_state=seed, n_init="auto")
    labels = kmeans.fit_predict(vectors)

    selected = []
    for cluster_id in range(n_samples):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        centroid = kmeans.cluster_centers_[cluster_id]
        dists = np.linalg.norm(vectors[cluster_indices] - centroid, axis=1)
        selected.append(int(cluster_indices[np.argmin(dists)]))

    return selected


def fork_dataset_subset(
    client: Client,
    dataset_version: DatasetVersion,
    selected_data_ids: list[str],
    new_version_name: str,
    with_annotations: bool,
    with_tags: bool,
) -> DatasetVersion:
    """Fork the source DatasetVersion keeping only the selected assets.

    selected_data_ids are (Data) UUIDs (see fetch_embeddings): they are
    resolved into (Asset) of this DatasetVersion via list_assets(data_ids=...).
    """
    parent_dataset = client.get_dataset(name=dataset_version.name)
    try:
        parent_dataset.get_version(new_version_name)
    except ResourceNotFoundError:
        pass
    else:
        raise ValueError(
            f"A dataset version named '{new_version_name}' already exists on "
            f"dataset '{dataset_version.name}'. Delete it or choose a different "
            "'new_version_name' input, then re-run the pipeline."
        )

    print(f"Resolving assets for {len(selected_data_ids)} selected data ids...")
    selected_assets = dataset_version.list_assets(data_ids=selected_data_ids)

    if len(selected_assets) != len(selected_data_ids):
        print(
            f"Warning: {len(selected_data_ids)} data ids selected but "
            f"{len(selected_assets)} assets found in this DatasetVersion."
        )

    print(f"Creating new DatasetVersion '{new_version_name}'...")
    try:
        new_version, _ = dataset_version.fork(
            version=new_version_name,
            description=(
                f"Diverse subset ({len(selected_assets)} images) extracted from "
                f"version '{dataset_version.version}' via embeddings"
            ),
            assets=selected_assets,
            with_tags=with_tags,
            with_labels=with_annotations,
            with_annotations=with_annotations,
            wait=True,
        )
    except ResourceConflictError as e:
        raise ValueError(
            f"A dataset version named '{new_version_name}' already exists on "
            f"dataset '{dataset_version.name}'. Delete it or choose a different "
            "'new_version_name' input, then re-run the pipeline."
        ) from e

    print(
        f"New DatasetVersion created: {new_version.name}/{new_version.version} "
        f"(id: {new_version.id})"
    )
    return new_version
