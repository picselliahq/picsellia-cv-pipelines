# Diversified Dataset Extractor

**Build a new, visually diverse dataset version by dropping near-duplicate images.**

This Picsellia processing embeds every image of an input dataset version with
an OpenCLIP model and keeps only those that are visually different enough from
the ones already kept. The result is a brand-new dataset version containing a
diverse, de-duplicated subset of the original images — ideal for trimming
redundancy before labeling or training.

---

## How it works

The selection is a **greedy, streaming nearest-neighbour filter** — not an
all-pairs comparison:

1. Images are streamed from the input dataset version in batches of 10
   (in dataset order).
2. Each image is fetched from its URL, EXIF-rotated, converted to RGB, and
   encoded into an embedding vector by the OpenCLIP model.
3. A [KD-tree](https://en.wikipedia.org/wiki/K-d_tree) holds the embeddings of
   all images kept so far:
   - The **first** image is always kept and seeds the tree.
   - For every later image, the pipeline queries the KD-tree for the
     **distance to the nearest already-kept embedding**. If that distance is
     **greater than `distance_threshold`**, the image is kept (added to the
     tree and uploaded); otherwise it is skipped as "too similar".
4. Kept images are added to the **output dataset version** in batches.

Two consequences worth knowing:

- **Order matters.** Because selection is sequential, the kept subset depends
  on the order images are streamed. The same threshold can yield slightly
  different subsets on reordered data.
- **Distance is Euclidean (L2) on raw CLIP embeddings**, not cosine
  similarity. That is why a sensible `distance_threshold` is a small positive
  number like `3–10` rather than a `0–1` ratio. The right value depends on the
  chosen architecture/weights, so tune it empirically.

---

## Input / Output

| | |
|---|---|
| **Processing type** | `DATASET_VERSION_CREATION` |
| **Input** | An existing dataset version (any inference type). |
| **Output** | A **new** dataset version containing the diverse image subset. Its inference type is copied from the input; its description records the source version and the `distance_threshold` used. |
| **Compute** | Runs on GPU if available, otherwise CPU. The shipped `config.toml` requests `cpu = 8`, `gpu = 0`, so on Picsellia it runs on CPU by default. |

> ⚠️ **Only images are copied, not annotations.** The processing calls
> `add_data` to put the selected images into the output version — it does not
> transfer existing annotations. Plan to (re-)annotate the output version, or
> use it as a curated pool for labeling.

---

## Parameters

Set these in the Picsellia processing job configuration (or the
`[parameters]` block of a local run config). Each parameter accepts several
alias keys.

### `distance_threshold`
Minimum embedding distance an image must have from every already-kept image to
be selected.

- **Aliases**: `distance`, `dist`, `dist_threshold`, `distance_threshold`
- **Type**: integer · **Default**: `5` · **Range**: `> 0`
- **Lower** → stricter → fewer, more diverse images.
- **Higher** → looser → more images kept.

```
distance_threshold = 3   →  very diverse (fewest images)
distance_threshold = 5   →  balanced (default)
distance_threshold = 8   →  lenient (most images kept)
```

Tuning: too few images kept → raise it; not diverse enough → lower it.

### `embedding_model`
Embedding backend to use.

- **Aliases**: `embedding_model`, `model`
- **Type**: string · **Default**: `"openclip"`
- **Supported**: `"openclip"` (only). Any other value raises an error.

### `model_architecture`
OpenCLIP architecture used to embed images. Validated at startup against
`open_clip.list_models()`.

- **Aliases**: `model_architecture`, `architecture`
- **Type**: string · **Default**: `"ViT-B-16-plus-240"`
- Common choices: `ViT-B-32` (fastest), `ViT-B-16-plus-240` (balanced),
  `ViT-L-14` (highest quality, slowest).

### `pretrained_weights`
Pretrained weight tag for the chosen architecture. Validated at startup against
`open_clip.pretrained.list_pretrained_tags_by_model(model_architecture)`.

- **Aliases**: `pretrained_weights`, `weights`
- **Type**: string · **Default**: `"laion400m_e32"`
- The valid tags depend on the architecture (e.g. `openai`,
  `laion2b_s34b_b79k`). An invalid tag fails fast with the list of valid ones.

> The architecture and weights must be a compatible pair. If you change one,
> check the other is valid for it — the pipeline validates both before running.

---

## Running on Picsellia

1. Open the input dataset version you want to diversify.
2. Launch the **Diversified Dataset Extractor** processing on it.
3. Set parameters (all optional — defaults apply otherwise):
   ```toml
   distance_threshold = 5
   embedding_model = "openclip"
   model_architecture = "ViT-B-16-plus-240"
   pretrained_weights = "laion400m_e32"
   ```
4. Run it. A new dataset version is created with the selected images; its
   description notes the source version and threshold.
5. Review the result, then annotate it or send it to labeling.

**Tip:** start from the defaults on a sample, check how many images survive,
then adjust `distance_threshold` up or down.

---

## Running locally

```bash
cd pipelines/diversified_dataset_extractor
uv sync   # first time only
python pipeline.py --mode local --config-file runs/<your_run_config>.toml
```

The local run config needs the input/output dataset version IDs plus the
`[parameters]` block shown above.

---

## Pipeline structure

`pipeline.py` wires four steps (`steps.py`):

1. **`load_coco_datasets(skip_asset_listing=True)`** — loads the input/output
   dataset versions.
2. **`validate_data`** — checks the input dataset
   (`utils/data_validator.py`).
3. **`validate_weights`** — verifies `model_architecture` and
   `pretrained_weights` are valid OpenCLIP values
   (`utils/model_validator.py`).
4. **`load_model`** — builds the OpenCLIP embedding model
   (`utils/model_loader.py`).
5. **`process`** — runs `DiversifiedDataExtractorProcessing`
   (`utils/processing.py`): the streaming KD-tree selection described above.

```
pipelines/diversified_dataset_extractor/
├── pipeline.py              # Context + step wiring.
├── steps.py                 # validate_data / validate_weights / load_model / process.
├── utils/
│   ├── parameters.py        # ProcessingDiversifiedDataExtractorParameters.
│   ├── data_validator.py    # Input dataset checks.
│   ├── model_validator.py   # Architecture / weights validation.
│   ├── model_loader.py      # OpenCLIP embedding model wrapper.
│   └── processing.py        # Greedy KD-tree diversity selection.
├── config.toml              # Metadata, parameters class, Docker/compute spec.
├── pyproject.toml           # Dependencies (open_clip, torch, scipy, ...).
└── Dockerfile
```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Too few images kept | `distance_threshold` too low → raise it. |
| Not diverse enough / too many kept | `distance_threshold` too high → lower it. |
| Slow processing | Use a lighter `model_architecture` (e.g. `ViT-B-32`), or run with GPU. |
| `model ... is not supported yet` | `embedding_model` must be `openclip`. |
| Error listing valid weights/architectures | The architecture/weights pair is invalid — use a value from the printed valid list. |
| Output has no annotations | Expected — only images are copied. (Re-)annotate the new version. |
| Some images skipped as "errors" | Image fetch/decoding failed (network or corrupt file); these are counted separately and retried twice. |

---

## Notes

- Supported embedding backend: **OpenCLIP only**.
- Selection is **order-dependent** and uses **L2 distance on raw CLIP
  embeddings**.
- Batch size is fixed at 10 images.
- OpenCLIP resources: [OpenCLIP on GitHub](https://github.com/mlfoundations/open_clip).

**Type**: Dataset Version Creation · **Supported input types**: all.
