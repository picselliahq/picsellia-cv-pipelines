# Train / Test / Val Split Pipeline

**Automatically split a dataset into train, test, and validation subsets.**

This Picsellia pipeline divides a dataset version into separate train, test, and val forks based on configurable ratios. It handles asset tagging, empty annotation creation, and naming conflicts automatically.

## What You'll Get

After running this pipeline, your Picsellia workspace will contain:
- ✅ A `{version}_train` dataset version with training assets
- ✅ A `{version}_test` dataset version with test assets (if ratio > 0)
- ✅ A `{version}_val` dataset version with validation assets (if ratio > 0)
- ✅ Optional asset tags (`train`, `test`, `val`) on the source dataset
- ✅ All annotations and labels preserved in each fork

---

## Inputs vs Parameters

| Mechanism | Set when | Examples |
|-----------|----------|---------|
| **Inputs** | Launching the job on Picsellia | dataset to split, train/test/val ratios |
| **Parameters** | Configuring the processing | asset tagging, empty annotation handling |

---

## 📥 Inputs Reference

Inputs are configured when launching the processing job from the Picsellia platform.

### `input`
**What it does**: The dataset version to split.

**Type**: Dataset Version
**Required**: Yes

Select the dataset version to divide from your Picsellia workspace.

---

### `ratio_train`
**What it does**: Proportion of assets assigned to the training set.

**Type**: Number
**Required**: Yes

**Constraint**: `ratio_train + ratio_test + ratio_val` must equal `1.0`. `ratio_train` must be greater than `0`.

**Example**: `0.7` (70% of assets go to training)

---

### `ratio_test`
**What it does**: Proportion of assets assigned to the test set.

**Type**: Number
**Required**: Yes

Set to `0` to skip creating a test split.

**Example**: `0.15` (15% of assets go to test)

---

### `ratio_val`
**What it does**: Proportion of assets assigned to the validation set.

**Type**: Number
**Required**: Yes

Set to `0` to skip creating a validation split.

**Example**: `0.15` (15% of assets go to validation)

---

## 📋 Parameters Reference

### `embed_asset_without_annotation`
**What it does**: Creates an empty annotation for every asset that has no annotation before splitting.

**Type**: Boolean
**Default**: `true`

**When to use**:
- `true` — ensures all assets are included in the split even if unannotated; useful for keeping unlabeled images in the dataset
- `false` — assets without annotations are split as-is without any modification

**Note**: The dataset type must be configured (not `NOT_CONFIGURED`) to create empty annotations.

---

### `add_asset_tags`
**What it does**: Adds `train`, `test`, and `val` asset tags to the corresponding assets in the source dataset version.

**Type**: Boolean
**Default**: `true`

**When to use**:
- `true` — useful for tracking which split each asset belongs to directly in the source dataset
- `false` — skips tag creation; cleaner if you only want the forked versions

---

## Quick Start Guide

### 🎯 Standard 70 / 15 / 15 Split

**Inputs** (set when launching the job):

| Input | Value |
|-------|-------|
| `input` | *(select your dataset version)* |
| `ratio_train` | `0.7` |
| `ratio_test` | `0.15` |
| `ratio_val` | `0.15` |

**Parameters**:
```toml
[parameters]
embed_asset_without_annotation = true
add_asset_tags = true
```

---

### 🎯 Train / Val Only (No Test)

| Input | Value |
|-------|-------|
| `ratio_train` | `0.8` |
| `ratio_test` | `0` |
| `ratio_val` | `0.2` |

**Parameters**:
```toml
[parameters]
embed_asset_without_annotation = true
add_asset_tags = true
```

Setting `ratio_test = 0` skips the test fork entirely.

---

### 🎯 Train / Test Only (No Val)

| Input | Value |
|-------|-------|
| `ratio_train` | `0.8` |
| `ratio_test` | `0.2` |
| `ratio_val` | `0` |

Setting `ratio_val = 0` skips the validation fork.

---

## 🎓 Real-World Configuration Examples

### Example 1: Small Dataset (< 500 images)

**Inputs**: `ratio_train = 0.8`, `ratio_test = 0.1`, `ratio_val = 0.1`

**Parameters**:
```toml
[parameters]
embed_asset_without_annotation = true
add_asset_tags = true
```

Larger train ratio to maximise training data on small datasets.

---

### Example 2: Large Dataset (> 10,000 images)

**Inputs**: `ratio_train = 0.9`, `ratio_test = 0.05`, `ratio_val = 0.05`

**Parameters**:
```toml
[parameters]
embed_asset_without_annotation = false
add_asset_tags = false
```

Smaller val/test ratios still give thousands of evaluation images; skip tagging for speed.

---

### Example 3: Training Only (No Evaluation Splits)

**Inputs**: `ratio_train = 1.0`, `ratio_test = 0`, `ratio_val = 0`

**Parameters**:
```toml
[parameters]
embed_asset_without_annotation = true
add_asset_tags = false
```

Creates only a `_train` fork — useful when evaluation is handled separately.

---

## 📊 Understanding Your Results

### Output Naming

The pipeline forks the source dataset version with these names:

```
Source version: "v1"

→ Dataset: <same dataset>  Version: "v1_train"
→ Dataset: <same dataset>  Version: "v1_test"   (if ratio_test > 0)
→ Dataset: <same dataset>  Version: "v1_val"    (if ratio_val > 0)
```

If a version name already exists, a timestamp is appended automatically:
```
v1_train_1714900000.123456
```

### Ratio Constraints

| Rule | Requirement |
|------|-------------|
| Ratios must sum to exactly `1.0` | `ratio_train + ratio_test + ratio_val = 1` |
| Train ratio must be positive | `ratio_train > 0` |
| Test and val can be zero | Skip their fork entirely |
| Minimum dataset size | At least 3 assets required |

---

## ❓ Troubleshooting Guide

### Pipeline fails — "less than 3 assets"
The source dataset version must contain at least 3 assets before splitting.

### Pipeline fails — "sum of ratios is not 1"
Verify that `ratio_train + ratio_test + ratio_val = 1.0` exactly. Floating point precision matters — use values like `0.7 + 0.15 + 0.15` rather than `0.333 + 0.333 + 0.334`.

### Version name conflict
The pipeline automatically appends a timestamp to avoid conflicts. No action needed.

### Empty annotation error
If `embed_asset_without_annotation = true` fails, ensure the dataset type is configured (not `NOT_CONFIGURED`) in Picsellia before running.

---

## 💡 Best Practices

1. **Choose ratios based on dataset size**:
   - Small (< 500): `0.8 / 0.1 / 0.1`
   - Medium (500-5000): `0.7 / 0.15 / 0.15`
   - Large (> 5000): `0.9 / 0.05 / 0.05`

2. **Keep the source dataset intact** — the pipeline forks assets, it does not move or delete them from the source version

3. **Use `add_asset_tags = true`** when you want to filter assets by split directly in the source dataset UI

4. **Stratify manually if needed** — this pipeline uses random splitting; for class-balanced splits, pre-filter your dataset before running

---

## 🚀 Getting Started Checklist

- [ ] Have a dataset version with at least 3 annotated assets in Picsellia
- [ ] Decide on split ratios (must sum to 1.0)
- [ ] Set `input` to your dataset version
- [ ] Set `ratio_train`, `ratio_test`, `ratio_val` inputs
- [ ] Configure `embed_asset_without_annotation` and `add_asset_tags` parameters
- [ ] Run the pipeline
- [ ] Verify the forked dataset versions in Picsellia

---

**Pipeline Version**: 1.0
**Type**: Pre-Annotation
**Supported Types**: Object Detection, Segmentation, Classification
