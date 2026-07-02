# Dataset Diversified Subset Pipeline

**Extract a diverse, representative subset of a dataset version from its Visual Search embeddings.**

This Picsellia pipeline pulls the embeddings already computed by the platform's Visual Search feature for a dataset version, selects a diverse subset of images from those embeddings using one of two algorithms, and forks the selected images into a new dataset version on the same parent dataset.

## What You'll Get

After running this pipeline, you'll have:
- A new dataset version containing only the selected, diverse subset of images
- A choice between two selection algorithms, controlled by the `algorithm` input
- Optionally, tags and annotations carried over from the source images

---

## Inputs Reference

Inputs are configured when launching the processing job from the Picsellia platform.

### `algorithm`
**What it does**: Selects which algorithm is used to pick the diverse subset.

**Type**: Text
**Required**: Yes
**Accepted values** (exactly two):

| Value | Algorithm | Behavior |
|-------|-----------|----------|
| `fps` | Farthest Point Sampling | Greedily picks images one at a time, always choosing the one farthest (in embedding space) from every image already selected. Maximizes coverage/spread across the whole dataset. |
| `kmeans` | K-means clustering | Groups all embeddings into `n_samples` clusters, then keeps the single image closest to each cluster's centroid. Picks one representative "typical" image per visual group. |

Any other value raises an error.

### `new_version_name`
**What it does**: Name of the new dataset version to create on the same parent dataset as the input.

**Type**: Text
**Required**: Yes

**Example**: `diverse-subset-200`

If a version with this name already exists on the dataset, the pipeline fails with a clear error instead of creating a duplicate — delete the existing version or pick a different name.

### `n_samples`
**What it does**: Number of images to select for the diverse subset.

**Type**: Number
**Required**: Yes

If `n_samples` is greater than or equal to the number of assets with embeddings, every asset is selected. If it's greater than the number of assets that have embeddings at all, the pipeline raises an error.

---

## Parameters Reference

### `embedder_key`
**What it does**: Which embedding model's vectors to use, if the dataset version has more than one indexed.

**Type**: String (optional)
**Default**: `None`

If the dataset version only has one embedding model indexed, this can be left unset. If several are available, the pipeline raises an error listing the available keys so you can set this parameter.

### `with_annotations`
**What it does**: Whether to copy labels and annotations from the source images into the new dataset version.

**Type**: Boolean
**Default**: `false`

### `with_tags`
**What it does**: Whether to copy asset tags from the source images into the new dataset version.

**Type**: Boolean
**Default**: `false`

### `seed`
**What it does**: Random seed used by both algorithms (the FPS starting point, and K-means initialization) for reproducible selection.

**Type**: Integer
**Default**: `0`

---

## How It Works

The pipeline runs three steps in sequence:

### Step 1 — `fetch_dataset_embeddings`
Fetches every indexed embedding vector for the target dataset version via the SDK (`count_embeddings`, `list_embeddings`). If several embedding models are indexed, `embedder_key` disambiguates which one to use.

### Step 2 — `select_diverse_subset`
Runs either Farthest Point Sampling or K-means clustering (per the `algorithm` input) on the fetched vectors to select `n_samples` diverse images.

### Step 3 — `create_subset_dataset_version`
Resolves the selected embeddings back to assets, then forks the source dataset version — keeping only the selected assets — into a new dataset version named `new_version_name`.

---

## Input Dataset Requirements

- The target dataset version must have embeddings computed, i.e. Visual Search must already be activated on it (`dataset_version.activate_visual_search()`). The pipeline raises a clear error if no embeddings are found.

---

## Quick Start

| Input | Example value |
|-------|---------------|
| `algorithm` | `fps` |
| `new_version_name` | `diverse-subset-200` |
| `n_samples` | `200` |

1. Activate Visual Search on the source dataset version if not already done.
2. Create a processing job on that dataset version.
3. Set the three inputs above.
4. Run — the new, diverse dataset version appears on the same parent dataset.

---

## Troubleshooting

### No embeddings found
Visual Search hasn't been activated on the source dataset version, or embedding computation hasn't finished yet. Activate it and wait for indexing to complete before running this pipeline.

### Several embedding models are available
The dataset version has more than one embedding model indexed. Set the `embedder_key` parameter to one of the keys listed in the error message.

### `n_samples` > number of assets with embeddings
Lower `n_samples`, or make sure Visual Search indexing has finished for all the assets you expect to select from.

### A dataset version with that name already exists
Delete the existing dataset version, or choose a different `new_version_name` input and re-run.

### Fewer images than requested with `kmeans`
Some clusters can end up empty for certain embedding distributions, so K-means may return fewer than `n_samples` images. The pipeline logs a warning when this happens; `fps` always returns exactly `n_samples` images.

---

**Pipeline Version**: 1.0
**Type**: Dataset Version Creation
**Selection Algorithms**: `fps` (Farthest Point Sampling), `kmeans` (K-means clustering)
