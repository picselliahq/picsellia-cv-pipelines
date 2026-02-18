# Shapes Cropper Pipeline

**Extract individual object crops from images based on bounding box / polygons annotations.**

This Picsellia pipeline creates a new dataset by cropping out objects from annotated images. Perfect for creating classification datasets from object detection or segmentation data, extracting specific objects for analysis, or preparing data for specialized models.

## What You'll Get

After running this pipeline, you'll have:
- ✅ A new dataset with cropped object images
- ✅ One image per bounding box / polygon annotation
- ✅ Images sized to the original bounding box / polygon dimensions
- ✅ Organized by label/class

## Quick Start Guide

### 🎯 Extract Single Class

**Goal**: Crop all instances of one object class.

```toml
[parameters]
label_name_to_extract = "person"
datalake = "default"
data_tag = "person_crops"
```

This will extract all "person" bounding boxes / polygons as individual images.

### 🎯 Extract for Classification

**Goal**: Create a classification dataset from detection annotations.

```toml
[parameters]
label_name_to_extract = "product"
datalake = "default"
data_tag = "product_classification"
fix_annotation = true
```

Perfect for building classification datasets from detection or segmentation data.

---

## 📋 Complete Parameter Reference

### 🔵 Essential Parameters

#### `label_name_to_extract`
**What it does**: Name of the label/class to extract crops from.

**Type**: String
**Default**: `"car"`
**Required**: Yes

**How it works**: Only bounding boxes / polygons with this exact label will be cropped and saved.

**Example**:
```toml
# Extract all persons
label_name_to_extract = "person"

# Extract all vehicles
label_name_to_extract = "car"

# Extract products
label_name_to_extract = "bottle"
```

**Important**: The label name must exactly match a label in your dataset.

---

#### `datalake`
**What it does**: Name of the datalake to store cropped images.

**Type**: String
**Default**: `"default"`

**Example**:
```toml
datalake = "default"

# Custom datalake
datalake = "object-crops"
```

---

#### `data_tag`
**What it does**: Tag to apply to cropped images for organization.

**Type**: String
**Default**: `"processed"`

**Example**:
```toml
data_tag = "processed"

# Descriptive tag
data_tag = "person_crops_2024"
```

---

#### `fix_annotation`
**What it does**: Attempt to fix invalid annotations during processing.

**Type**: Boolean
**Default**: `true`

**What it fixes**:
- Bounding boxes / polygons outside image boundaries
- Invalid coordinates
- Malformed annotations

**Example**:
```toml
# Fix issues automatically (recommended)
fix_annotation = true

# Strict mode - fail on invalid annotations
fix_annotation = false
```

---

## 🎓 Real-World Configuration Examples

### Example 1: Person Detection → Person Classification

**Goal**: Extract person crops for age/gender classification.

```toml
[parameters]
label_name_to_extract = "person"
datalake = "default"
data_tag = "person_crops_for_classification"
fix_annotation = true
```

**Result**: Individual person images ready for classification labeling.

---

### Example 2: Product Extraction from Shelf Images

**Goal**: Extract individual products from retail shelf photos.

```toml
[parameters]
label_name_to_extract = "product"
datalake = "retail-products"
data_tag = "individual_products"
fix_annotation = true
```

**Result**: Each product as a separate image for catalog or quality control.

---

### Example 3: Vehicle Analysis

**Goal**: Extract vehicle crops for make/model classification.

```toml
[parameters]
label_name_to_extract = "car"
datalake = "default"
data_tag = "vehicle_crops"
fix_annotation = true
```

**Result**: Individual vehicle images for detailed analysis.

---

### Example 4: Document Fields Extraction

**Goal**: Extract specific fields from document images.

```toml
[parameters]
label_name_to_extract = "signature"
datalake = "default"
data_tag = "signature_crops"
fix_annotation = true
```

**Result**: Isolated signature regions from documents.

---

### Example 5: Wildlife Camera Traps

**Goal**: Extract animal crops from camera trap images.

```toml
[parameters]
label_name_to_extract = "animal"
datalake = "wildlife"
data_tag = "animal_crops_2024"
fix_annotation = true
```

**Result**: Individual animal images from wide-angle camera trap photos.

---

## 📊 Understanding Your Results

### What Gets Created

For each bounding box / polygon with the specified label:
1. **Cropped image**: Extracted region from original image
2. **Original dimensions**: Crop size matches bounding box / polygon size
3. **Metadata preserved**: Links to original image

### Output Dataset Structure

```
Original image: street_scene.jpg
  └─ Contains 3 "person" bounding boxes / polygons

Output dataset:
  ├─ street_scene_person_001.jpg  (crop 1)
  ├─ street_scene_person_002.jpg  (crop 2)
  └─ street_scene_person_003.jpg  (crop 3)
```

### Crop Sizing

Crops maintain original bounding box / polygons dimensions:
- Small shapes (50x100) → Small crop (50x100)
- Large shapes (300x400) → Large crop (300x400)

**Note**: You may want to resize crops later for model training consistency.

---

## ❓ Troubleshooting Guide

### Issue: No Crops Created

**Check**:
1. Does `label_name_to_extract` exactly match a label in your dataset?
2. Are there any bounding boxes / polygons with that label?
3. Check pipeline logs for errors
4. Verify dataset has annotations

---

### Issue: Some Bounding Boxes / Polygons Skipped

**Possible causes**:
1. Bounding boxes outside image boundaries (set `fix_annotation = true`)
2. Invalid coordinates
3. Zero-area bounding boxes / polygons

**Solution**:
```toml
fix_annotation = true
```

---

### Issue: Crops Are Different Sizes

**This is expected**: Crops preserve original bounding box / polygons dimensions.

**Solution**: If you need uniform sizes, resize crops after extraction:
- Use image processing pipeline
- Resize during training
- Use padding to standardize

---

### Issue: Wrong Objects Extracted

**Check**:
1. Verify `label_name_to_extract` spelling
2. Check label names are correct in dataset
3. Review original annotations

---

## 💡 Best Practices

### 1. Verify Label Name

```toml
# Check exact label name in your dataset first
# Labels are case-sensitive!
label_name_to_extract = "Person"  # ❌ if dataset has "person"
label_name_to_extract = "person"  # ✅ exact match
```

### 2. Use Descriptive Tags

```toml
# Good tags
data_tag = "person_crops_retail_2024"
data_tag = "vehicle_front_view"

# Less helpful tags
data_tag = "crops"
data_tag = "processed"
```

### 3. Fix Annotations Automatically

```toml
fix_annotation = true  # Recommended for most cases
```

### 4. Common Workflow

1. **Object Detection or Segmentation** → Annotate with bounding boxes or polygons
2. **Shapes Cropper** → Extract individual objects
3. **Manual Review** → Review extracted crops
4. **Classification Labeling** → Add classification labels
5. **Train Classifier** → Train on cropped objects

### 5. Organize by Datalake

```toml
# Use different datalakes for different projects
datalake = "person-classification"
datalake = "product-catalog"
datalake = "vehicle-analysis"
```

---

## 🚀 Getting Started Checklist

- [ ] Have dataset with bounding box / polygon annotations
- [ ] Identify label to extract
- [ ] Verify label name exactly matches dataset
- [ ] Choose datalake and tag for organization
- [ ] Enable fix_annotation (recommended)
- [ ] Run pipeline
- [ ] Review extracted crops
- [ ] Use crops for downstream tasks

---

## 🔗 Related Pipelines

- **Dataset Tiler**: Alternative for large images
- **YOLOv8 Pre-Annotation**: Generate bounding boxes first
- **Albumentations Processing**: Augment cropped images

---

## 🎯 Use Cases

### Object Classification
Start with detection annotations → Extract crops → Label for classification → Train classifier

### Quality Control
Extract products → Review for defects → Automated QC system

### Data Analysis
Extract specific objects → Analyze characteristics → Generate insights

### Dataset Curation
Extract objects → Filter/sort → Build specialized datasets

---

## 📞 Support

**Need help?**
- Picsellia platform questions → Contact your Picsellia support team
- Pipeline configuration help → Refer to this guide

---

**Pipeline Version**: 1.0.4
**Type**: Dataset Version Creation
**Supported Types**: Object Detection
**Last Updated**: 2026-01-08
