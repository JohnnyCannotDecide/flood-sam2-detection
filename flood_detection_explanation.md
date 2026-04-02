# flood_detection.py — Deep Dive Explanation

## Overview

This is a **flood detection and early warning system** that processes multi-temporal SAR (Synthetic Aperture Radar) satellite imagery to detect water bodies, track changes over time, and generate risk reports. It's a comprehensive remote sensing pipeline combining computer vision, deep learning (SAM2), and LLM-assisted analysis.

---

## Architecture & Data Flow

```
Input TIF Files
      |
      v
+-----------------------------------------------------------+
|  Image Preprocessing (load, enhance, detect nodata)        |
|                        |                                    |
|                        v                                    |
|  Water Probability Map (guided filter, Otsu thresholding) |
|                        |                                    |
|                        v                                    |
|  SAM2 Segmentation (sliding window with cosine blending)   |
|          or                                                 |
|  Probabilistic Segmentation (fallback)                      |
|                        |                                    |
|                        v                                    |
|  Post-Processing (morphological ops, connected components) |
|                        |                                    |
|                        v                                    |
|  Feature Extraction (contours, shape metrics)              |
|                        |                                    |
|                        v                                    |
|  Change Detection (persistent/receding/new water)          |
|                        |                                    |
|                        v                                    |
|  Quality Assessment (IoU, F1, area accuracy)               |
|                        |                                    |
|                        v                                    |
|  LLM Expert Review (quality + warning analysis)            |
|                        |                                    |
|                        v                                    |
|  Outputs: Visualizations, JSON report, Interactive website|
+-----------------------------------------------------------+
```

---

## Key Functional Sections

### 1. Configuration & Paths (lines 33-57)

- Sets up file paths for SAM2 (Segment Anything Model 2)
- Default output directory: `output/`
- Pixel area for area calculation: 25 m²/pixel
- LLM configuration (OpenRouter API with fallback model)
- Window parameters: 512×512 with 128-pixel overlap for SAM2

### 2. Mask Caching (lines 97-136)

**Purpose:** Avoid re-running expensive SAM2 segmentation on the same images

```python
# Cache file format: {label}_mask_cache.npz
# Stores: water_mask, nodata_mask, area_km2 (compressed)
```

- Saves compressed numpy arrays (`.npz`) to disk
- On subsequent runs, checks if cache exists and shape matches
- **Significant performance optimization** for re-analysis

**Cache functions:**
- `mask_cache_path()`: Generates cache file path
- `save_mask_cache()`: Persists segmentation results
- `load_mask_cache()`: Attempts to load cached results with validation

### 3. Image Preprocessing (lines 142-240)

**`load_all_images()`**:
- Loads multiple TIF files using `rasterio`
- Aligns all images to smallest common dimensions
- Extracts metadata (transform, CRS, dimensions)

**`detect_nodata()`**:
- Uses flood-fill algorithm (`cv2.floodFill`) to identify image edges with no data
- Scans for zeros near borders and fills contiguous regions
- Identifies radar "blind spots" in the imagery

**`enhance()`**:
Applies triple enhancement pipeline:
1. **Percentile stretching** (2%-98%): Normalizes dynamic range
2. **Log transformation** (`log(x+1)`): Enhances dark regions
3. **Gamma correction** (power=0.35): Improves contrast

**`guided_filter()`**:
- Edge-preserving smoothing using box filters
- Formula: `output = a * I + b` where `a` handles edges, `b` handles smooth areas

**`build_water_prob()`**:
Core water detection algorithm:
1. Applies guiding filter to enhanced image
2. Uses **Otsu thresholding** on filtered image
3. Applies stricter threshold (60% of Otsu value, max 45)
4. Detects **deep water**: values ≤8 with backscatter >0
5. Excludes **shadows** using local gradient analysis
   - Calculates morphological gradient (dilation - erosion)
   - Shadows have low gradient values
6. Generates probability map with three levels:
   - `1.0` = definitive water (deep water or high-confidence)
   - `0.2` = shadow/ambiguous water
   - `0.0` = land or no data
7. Applies Gaussian blur for smooth probability distribution

### 4. SAM2 Segmentation (lines 272-359)

**Why SAM2?** Meta's Segment Anything Model 2 provides state-of-the-art promptable segmentation with minimal training data.

**How it works:**

1. **Sliding window approach**:
   - Window size: 512×512 pixels
   - Overlap: 128 pixels
   - Ensures coverage of full image

2. **Cosine blending**:
   - Each pixel weighted by cosine window
   - Avoids tile artifacts at boundaries
   - Formula: `weight = cos²(distance / size * π)`

3. **Point prompts**: Samples points from probability map:
   - 6 foreground points (high prob > 0.5)
   - 4 background points (low prob < 0.15)
   - Minimum distance constraint: 20 pixels between samples

4. **SAM2 prediction**:
   - `SAM2ImagePredictor.set_image()` on grayscale→RGB converted patch
   - `predictor.predict()` with multimask_output=True
   - Selects best mask by score

5. **Voting mechanism**:
   - Each window's result weighted by prediction confidence
   - Accumulates votes across all windows
   - Final mask = votes / total_weights > 0.5

6. **Fallback**: If SAM unavailable or score < 0.3, uses probability threshold directly

**Key parameters:**
- `WINDOW_SIZE = 512`
- `OVERLAP = 128`
- `N_FG = 6` (foreground points)
- `N_BG = 4` (background points)
- `SAM2_SCORE_MIN = 0.3`

**`init_sam2_predictor()`**:
- Checks for SAM2 installation and checkpoint
- Builds model with hybrid architecture (Hiera Small)
- Detects CUDA availability for GPU acceleration

### 5. Post-Processing (lines 325-336)

```python
1. Morphological closing (9×9 kernel)
   → Fills small gaps in water bodies

2. Morphological opening (3×3 kernel)
   → Removes single-pixel speckle noise

3. Connected components analysis
   → Identifies separate water bodies
   → Filters by minimum area (300 pixels)

4. Mask invalid regions
   → Excludes nodata areas from final water mask
```

**Purpose:** Removes noise while preserving true water features.

### 6. Feature Extraction (lines 370-407)

Extracts shape statistics for each disconnected water body:

Using OpenCV contour analysis:
- **Contours and areas**: Detected via `cv2.findContours()`
- **Perimeter**: Calculated with `cv2.arcLength()`
- **Bounding boxes**: Minimum rectangle enclosing each contour
- **Centroids**: Using image moments (`cv2.moments()`)

**Shape metrics:**
- **Compactness** (circularity): `4πA/P²`
  - Measures how close shape is to a circle
  - Range: 0 (irregular) → 1 (perfect circle)
- **Elongation**: `max(width, height) / min(width, height)`
  - Aspect ratio of minimum-area rotating rectangle
  - Higher = more elongated (e.g., rivers)

**Returns:**
```python
{
    "contour_count": N,
    "components_count": M,
    "largest_component_px": int,
    "smallest_component_px": int,
    "mean_compactness": float,
    "mean_elongation": float,
    "top_contours": [top 20 contours with full stats]
}
```

### 7. Change Detection (lines 601-649)

Compares consecutive time steps to detect water movement:

```
T1 → T2 categories:
┌─────────────────────────────────────┐
│  Persistent:   water → water  (blue)│
│  Receding:     water → land   (red) │
│  New:          land → water    (green)│
└─────────────────────────────────────┘
```

**Algorithm:**
1. Masks aligned to valid intersection region (pixels valid in both)
2. For each consecutive pair:
   - `persistent = mask_T1 & mask_T2`
   - `receding = mask_T1 & ~mask_T2`
   - `new = ~mask_T1 & mask_T2`
3. Calculates area in km² for each category
4. Generates RGB overlay visualization:
   - Blue: persistent water (80% weight)
   - Red: receding water (red tint)
   - Green: new water (green tint)
   - Gray: nodata regions

### 8. Quality Assessment (lines 413-459)

Computes segmentation accuracy metrics:

**Metric definitions:**
- **IoU** (Intersection over Union): `TP / (TP + FP + FN)`
- **Precision**: `TP / (TP + FP)` (proportion of detected water that is correct)
- **Recall**: `TP / (TP + FN)` (proportion of true water detected)
- **F1 Score**: `2 × (Precision × Recall) / (Precision + Recall)`
- **Area Accuracy**: `1 - |pred_area - ref_area| / ref_area`

**Input handling:**
- Ground truth masks loaded if provided
- Otherwise uses high-threshold probability map as pseudo-reference
- Parallel evaluation via `ThreadPoolExecutor` (up to 8 workers)

**Aggregation:**
Computes mean across all input images for each metric.

### 9. LLM Integration (lines 462-549)

**Two expert analysis modes:**

#### 1. Quality Review (`llm_quality_review`)

**Purpose:** Evaluates if segmentation meets quality standards

**Input:** Quality metrics (IoU, F1, Precision, Recall, area accuracy)

**LLM Prompt Structure:**
```
System: "你是遥感分割质量审查专家，仅输出JSON对象。"

User: {
  "task": "评估分割质量是否达标",
  "metrics": {IoU, F1, Precision, Recall, area_accuracy},
  "required_keys": ["verdict", "confidence", "reasons", "suggestions"]
}
```

**Output:**
```json
{
  "mode": "external_llm",
  "verdict": "通过" | "待优化",
  "confidence": 0.0-1.0,
  "reasons": ["IoU=0.821", "F1=0.789", ...],
  "suggestions": ["提高前景提示点数量", ...]
}
```

**Fallback:** Rule-based scoring when LLM unavailable:
- Score = 0.35×IoU + 0.35×F1 + 0.15×Precision + 0.15×Recall
- Verdict = "通过" if score ≥ 0.75 and area_accuracy ≥ 0.75

#### 2. Warning Expert (`llm_warning_expert`)

**Purpose:** Generates flood risk analysis and recommendations

**Input:** Full warning report stub (trends, risk assessment, etc.)

**LLM Prompt Structure:**
```
System: "你是专业洪水预警分析专家，仅输出JSON对象。"

User: {
  "task": "根据输入生成洪水风险分析建议",
  "input": {...},
  "required_keys": ["trend", "risk", "warning_level", "impact_scope", "recommendations"]
}
```

**Output:**
```json
{
  "mode": "external_llm",
  "trend": "increasing",
  "risk": "持续扩张，高风险...",
  "warning_level": "高",
  "impact_scope": "约XX平方公里",
  "recommendations": [...]
}
```

**API Configuration:**
- Provider: OpenRouter
- Model: Google Gemini 2.5 Pro (configurable)
- Fallback: Rule-based proxy analysis

### 10. Warning Report Generation (lines 689-766)

Computes comprehensive risk assessment and generates structured report.

**Risk Score Formula:**
```python
score = min(1.0,
    0.45 × growth_ratio +                    # Expansion rate
    0.25 × (new_water / total_area) +        # New water proportion
    0.30 × (1 - area_accuracy)               # Quality penalty
)
```

**Warning Levels:**
| Score | Level |
|-------|-------|
| ≥ 0.7 | High (高) |
| ≥ 0.4 | Medium (中) |
| < 0.4 | Low (低) |

**Impact Prediction:**
```python
impact_km2 = current_area × (1.15 + score × 0.5)
confidence = 0.55 + (1 - area_accuracy) × 0.35
```

**JSON Structure (ver 1.0):**
```json
{
  "schema_version": "1.0",
  "project": "flood_detection",
  "input_summary": {image_count, files, branch},
  "water_segmentation_results": [...],
  "change_analysis": [...],
  "quality_assurance": {metrics_summary, llm_quality_review},
  "trend": {direction, areas_km2, delta_km2, ...},
  "risk_assessment": {risk_score, warning_level, summary},
  "impact_scope_prediction": {estimated_impact_km2, confidence},
  "decision_support": {expert_opinion, warning_recommendations, response_actions},
  "expert_model_output": {...}
}
```

### 11. Website Generation (lines 772-791)

Generates interactive HTML dashboard via `flood_web.write_dashboard()`.

**Features:**
- Before/after slider comparison for time-series analysis
- Area statistics visualization
- Change detection overlays with color coding
- Quality metrics dashboard
- Responsive design for various screen sizes

**Data passed:**
```python
{
    "labels": [...],
    "areas": [...],
    "result_images": [...],
    "seg_images": [...],
    "prob_images": [...],
    "change_images": [...],
    "change_stats": [{persistent, receding, new}, ...],
    "quality": [...],
    "report": {...}
}
```

### 12. Main Pipeline (`run_pipeline`, lines 797-869)

**Orchestrates the entire workflow:**

```
1. Load and align images (rasterio)
2. Initialize SAM2 predictor (if enabled)
3. For each image:
   ├─ Detect nodata regions
   ├─ Try loading cache (unless force_rerun)
   ├─ Build water probability map
   ├─ Run SAM2 segmentation (or fallback)
   ├─ Save visualization results
   └─ Cache segmentation results
4. Compute valid intersection region
5. If ≥2 images:
   └─ Run change detection with visualization
6. Generate summary visualization (2×n grid)
7. Evaluate quality metrics (parallel)
8. Run LLM quality review
9. Build flood warning report with risk assessment
10. Generate interactive HTML website
11. Print final summary to console
```

**Key behaviors:**
- Graceful degradation when SAM2 unavailable
- Automatic caching for speed
- Parallel quality evaluation
- Multi-language support (Chinese labels)

---

## Key Code Patterns & Design Decisions

| Pattern | Purpose | Example |
|---------|---------|---------|
| **Context manager** | Temporarily change directory for imports | `with switch_dir(SAM2_ROOT):` |
| **Global state** | Shared area reference per pipeline | `global PIXEL_AREA_M2` |
| **Fallback semantics** | Graceful degradation | SAM2 → probability map |
| **Caching strategy** | Per-label compressed storage | `{label}_mask_cache.npz` |
| **Cosine window** | Avoids tile artifacts | `win = outer(hanning(t), hanning(t))` |
| **Parallel processing** | Efficient metric computation | `ThreadPoolExecutor(max_workers=8)` |
| **LLM dual mode** | External + rule-based fallback | `external or llm_quality_review()` |

---

## Dependencies

```python
# Core processing
cv2              # OpenCV (image processing, morphology)
numpy            # Numerical arrays
rasterio         # GeoTIFF I/O with metadata

# Visualization
matplotlib       # Plotting and image display

# Deep Learning
torch            # PyTorch (SAM2 backend)
sam2             # Segment Anything Model 2

# LLM (optional)
langchain_openai # OpenRouter API integration

# Local modules
flood_web        # HTML dashboard generator
```

---

## SAM2 Model Details

**Architecture:** Hierarchical Masked Autoencoder (Hiera Small)
-Checkpoint: `sam2.1_hiera_small.pt`
- Config: `sam2.1_hiera_s.yaml`

**Why Hiera?**
- Efficient hierarchical vision transformer
- Good performance with smaller model size
- Suitable for inference on CPU/GPU

**Device detection:**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## Usage Examples

### Command Line Interface

```bash
# Basic usage with directory scan
python flood_detection.py --input-dir ./satellite_data

# Specify files and labels
python flood_detection.py --input-files d27.tif d28.tif d29.tif --labels D27 D28 D29

# Custom pixel area (resolution calibration)
python flood_detection.py --pixel-area-m2 25

# Disable SAM2 (use probability map only)
python flood_detection.py --disable-sam2

# Force re-segmentation (ignore cache)
python flood_detection.py --force-rerun

# Custom output directory
python flood_detection.py --out-dir ./results/flood_analysis

# With ground truth for evaluation
python flood_detection.py --ground-truth-files gt1.tif gt2.tif
```

### Python API

```python
from flood_detection import process_uploaded_images

payload = {
    "input_files": ["path/to/image1.tif", "path/to/image2.tif"],
    "labels": ["Time1", "Time2"],
    "pixel_area_m2": 25.0,
    "enable_sam2": True,
    "force_rerun": False,
    "out_dir": "./output"
}

result = process_uploaded_images(payload)
# Returns: {"output_dir": ..., "warning_report": ..., "site_index": ...}
```

---

## CLI Arguments Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input-files` | list | None | Specific TIF files to process |
| `--input-dir` | str | ROOT_DIR | Directory to scan for .tif/.tiff files |
| `--labels` | list | None | Custom labels for each image |
| `--ground-truth-files` | list | None | Reference masks for quality evaluation |
| `--out-dir` | str | "./output" | Output directory for results |
| `--pixel-area-m2` | float | 25.0 | Area represented by one pixel (m²) |
| `--disable-sam2` | flag | False | Skip SAM2, use probability segmentation only |
| `--force-rerun` | flag | False | Ignore cache, re-segment everything |

---

## Output Files

| File | Format | Description |
|------|--------|-------------|
| `{label}_mask_cache.npz` | NumPy compressed | Cached water mask (reusable) |
| `{label}_prob_map.png` | Image | Water probability visualization |
| `{label}_seg_only.png` | Image | Water overlay only (for slider) |
| `{label}_water_result.png` | Image | Full visualization with legend |
| `change_{label1}_{label2}.png` | Image | Change detection overlay |
| `summary.png` | Image | 2×n grid of all results |
| `warning_report.json` | JSON | Structured analysis report |
| `index.html` | HTML | Interactive dashboard |

---

## Workflow Diagram

```
User Input
    |
    v
+---------------+
| Flood Detection |
+---------------+
    |
    +---> Load TIF Files
    |      |
    |      +---> Detect NoData
    |      +---> Enhance Image
    |      +---> Build Probability Map
    *      |
    |      v
    +---> SAM2 Segmentation?
    |     /     \
    |   Yes      No
    *   |         |
    |   v         v
    +--<--- Save Cache ----<---+
    |                        |
    +---> Extract Features   |
    |      |                 |
    +---> Change Detection <-+
    |      |
    +---> Quality Metrics
    |      |
    +---> LLM Review
    |      |
    +---> Risk Assessment
    |      |
    +---> Generate Report
    |      |
    +---> Build Website
    |
    v
Output Directory
```

---

## Performance Characteristics

| Operation | Notes |
|-----------|-------|
| SAM2 inference | ~0.5-2 sec per 512×512 window (GPU) |
| Probability map | ~0.1-0.5 sec per image |
| Cache hit | Near-instant (np.load + shape check) |
| Full pipeline (3 images, SAM2) | ~2-10 minutes depending on size |
| Without cache | Each image re-segmented |

---

## Quality Control Points

1. **Cache validation**: Shape must match to prevent misuse
2. **Minimum area**: 300 pixels to filter noise
3. **SAM2 score threshold**: 0.3 minimum confidence
4. **Valid region intersection**: Ensures comparable analysis
5. **Ground truth fallback**: Pseudo-reference if GT unavailable
6. **LLM sanity checks**: Validated JSON output parsing

---

This is a production-grade remote sensing pipeline that balances:
- **Accuracy**: SAM2 + probability fusion
- **Efficiency**: Caching + parallel processing
- **Explainability**: Visualizations + LLM analysis
- **Robustness**: Fallbacks at every critical step
