# Shapes Cropper Pipeline

**Extract individual object crops from images based on bounding box / polygons annotations.**

This Picsellia pipeline creates a new dataset by cropping out objects from annotated images. Perfect for creating classification datasets from object detection or segmentation data, extracting specific objects for analysis, or preparing data for specialized models.

## What You'll Get

After running this pipeline, you'll have:
- A new dataset with cropped object images
- One image per bounding box / polygon annotation for each extracted label
- Images sized to the original bounding box / polygon dimensions
- Support for extracting multiple labels in a single run

## Quick Start Guide

### Extract a Single Class

**Goal**: Crop all instances of one object class.

```toml
[inputs]
label_name_to_extract = "person"
datalake = "<your-datalake-id>"

[parameters]
data_tag = "person_crops"
```

### Extract Multiple Classes

**Goal**: Crop all instances of several object classes in one run.

```toml
[inputs]
label_name_to_extract = "car,truck,bus"
datalake = "<your-datalake-id>"

[parameters]
data_tag = "vehicle_crops"
fix_annotation = true
```

Each label is extracted and preserved with its own category in the output dataset.

---

## Complete Reference

### Inputs

Inputs are user-defined values selected at pipeline launch time (e.g. via the Picsellia UI or CLI).

---

#### `label_name_to_extract`
**What it does**: Name(s) of the label(s) to extract crops from. Accepts a single label or a comma-separated list.

**Type**: Text
**Required**: Yes

**Examples**:
```
person
car,truck,bus
product
```

All provided labels must exist in the source dataset's labelmap. Labels are case-sensitive.

---

#### `datalake`
**What it does**: The datalake where cropped images will be stored.

**Type**: Datalake
**Required**: Yes

Select the target datalake from the Picsellia UI when launching the pipeline.

---

### Parameters

Parameters control the processing behaviour and can be set in the pipeline configuration.

---

#### `data_tag`
**What it does**: Tag applied to cropped images in the datalake for organization.

**Type**: String
**Default**: `"processed"`

**Example**:
```toml
data_tag = "vehicle_crops_2024"
```

---

#### `fix_annotation`
**What it does**: Automatically correct invalid or out-of-bounds annotations instead of failing.

**Type**: Boolean
**Default**: `true`

**What it fixes**:
- Bounding boxes / polygons outside image boundaries
- Invalid coordinates
- Malformed annotations

```toml
# Fix issues automatically (recommended)
fix_annotation = true

# Strict mode - fail on invalid annotations
fix_annotation = false
```

---

## Configuration Examples

### Person Crops for Classification

**Goal**: Extract person crops for age/gender classification.

```toml
[inputs]
label_name_to_extract = "person"
datalake = "<your-datalake-id>"

[parameters]
data_tag = "person_crops_for_classification"
fix_annotation = true
```

---

### Multi-Class Vehicle Extraction

**Goal**: Extract all vehicle types in a single run.

```toml
[inputs]
label_name_to_extract = "car,truck,motorcycle,bus"
datalake = "<your-datalake-id>"

[parameters]
data_tag = "vehicle_crops"
fix_annotation = true
```

Each vehicle type is saved with its own category in the output COCO file.

---

### Product Extraction from Shelf Images

**Goal**: Extract individual products from retail shelf photos.

```toml
[inputs]
label_name_to_extract = "product"
datalake = "<your-datalake-id>"

[parameters]
data_tag = "individual_products"
fix_annotation = true
```

---

### Document Fields Extraction

**Goal**: Extract specific fields from document images.

```toml
[inputs]
label_name_to_extract = "signature,stamp"
datalake = "<your-datalake-id>"

[parameters]
data_tag = "document_fields"
fix_annotation = true
```

---

## Understanding Your Results

### Output Naming

Each cropped image is named using the pattern:

```
{data_id}_{label}_{x}_{y}_{width}_{height}{extension}
```

For example: `abc123_car_120_45_200_150.jpg`

### Output Dataset Structure

```
Original image: street_scene.jpg
  └─ Contains 2 "car" and 1 "person" bounding boxes

Output dataset (label_name_to_extract = "car,person"):
  ├─ abc123_car_120_45_200_150.jpg
  ├─ abc123_car_300_60_180_130.jpg
  └─ abc123_person_500_20_80_200.jpg
```

### Crop Sizing

Crops maintain original bounding box / polygon dimensions. You may want to resize them after extraction for model training consistency.

---

## Troubleshooting

### No Crops Created

1. Does `label_name_to_extract` exactly match a label in your dataset? Labels are case-sensitive.
2. Are there annotations with that label in the dataset?
3. Verify dataset has annotations.

---

### Some Annotations Skipped

**Possible causes**:
- Bounding boxes outside image boundaries → set `fix_annotation = true`
- Invalid or zero-area coordinates

---

### Crops Are Different Sizes

This is expected — crops preserve original bounding box / polygon dimensions. Resize crops after extraction if uniform sizes are needed.

---

## Best Practices

1. **Check label names** — Labels are case-sensitive. `"Person"` and `"person"` are different.
2. **Extract multiple labels at once** — Use `"car,truck"` instead of running the pipeline twice.
3. **Use descriptive tags** — `data_tag = "vehicle_crops_retail_2024"` is more useful than `"processed"`.
4. **Enable fix_annotation** — Recommended for most datasets.

### Typical Workflow

1. Annotate with bounding boxes or polygons (Object Detection or Segmentation)
2. Run Shapes Cropper to extract individual objects
3. Review extracted crops
4. Add classification labels
5. Train a classifier on the cropped objects

---

## Getting Started Checklist

- [ ] Dataset has bounding box / polygon annotations
- [ ] Identified the label(s) to extract
- [ ] Label names exactly match the dataset labelmap
- [ ] Target datalake selected
- [ ] `data_tag` set for organization
- [ ] `fix_annotation = true` (recommended)

---

## Related Pipelines

- **Dataset Tiler**: Alternative for large images
- **YOLOv8 Pre-Annotation**: Generate bounding boxes first
- **Albumentations Processing**: Augment cropped images

---

**Pipeline Version**: 1.0.6
**Type**: Dataset Version Creation
**Supported Annotation Types**: Object Detection, Segmentation
**Last Updated**: 2026-05-06
