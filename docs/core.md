# Core Functionality

This document describes the core functionality of the LFA analysis pipeline.

## Image Processing Pipeline

The core functionality is implemented in `src/core/` and includes:

### 1. Mask Generation (`generate_with_UNET.py`)

Main features:
- Background color detection
- Image padding
- Mask generation using UNET model
- Logging and visualization
- Batch processing support

```python
# Key functions
def get_background_color(image):
    """Detect background color from image edges"""
    # Implementation details...

def generate_mask(image_path, save_path):
    """Generate mask for a single image"""
    # Implementation details...
```

### 2. Mask Application (`apply_mask.py`)

Features:
- Mask overlay with red highlighting
- ROI extraction
- Result visualization
- Batch processing
- Mask coverage statistics

```python
def apply_mask_to_image(image_path, mask_path, output_path):
    """Apply mask to image and save the result"""
    # Implementation details...
```

### 3. Analysis and Visualization (`utils/analyze_masks.py`)

Features:
- Intensity analysis
- Statistical metrics
- Visualization of results
- Batch processing support
- Comprehensive reporting

## Background Color Detection

The pipeline includes intelligent background color detection:
1. Samples edges of the image
2. Identifies most common color
3. Uses detected color for padding
4. Maintains visual consistency

## Image Preprocessing

### Resizing and Padding
- Maintains aspect ratio
- Pads to 512x512
- Uses detected background color
- Handles various input sizes

### Normalization
- Zero mean
- Unit standard deviation
- Handles different input ranges

## Logging and Visualization

### Logging Features
- Transformation details
- Tensor statistics
- Error handling
- Progress tracking
- Performance metrics

### Visualization
- Original vs transformed images
- Mask overlays
- Processing results
- Error cases
- Analysis plots

## Usage Examples

### Generate Masks
```python
from src.core.generate_with_UNET import generate_mask

# Generate mask for single image
generate_mask(
    image_path="data/test_images/image.png",
    save_path="data/masks/mask.png"
)
```

### Apply Masks
```python
from src.core.apply_mask import apply_mask_to_image

# Apply mask to image
apply_mask_to_image(
    image_path="data/test_images/image.png",
    mask_path="data/masks/mask.png",
    output_path="data/masked_images/result.png"
)
```

### Analyze Results
```python
from src.utils.analyze_masks import analyze_mask

# Analyze a single mask
analyze_mask(
    test_image_path="data/test_images/image.png",
    mask_image_path="data/masks/mask.png",
    masked_image_path="data/masked_images/result.png",
    output_path="data/analysis/analysis.png"
)
```

## Error Handling

The pipeline includes robust error handling:
1. File existence checks
2. Image loading validation
3. Processing error recovery
4. Detailed error logging
5. Graceful failure handling

## Performance Considerations

1. **Memory Management**
   - GPU memory optimization
   - Batch processing
   - Resource cleanup
   - Efficient data loading

2. **Processing Speed**
   - Efficient image operations
   - Parallel processing where possible
   - Optimized data loading
   - Caching mechanisms

3. **Quality Control**
   - Input validation
   - Output verification
   - Consistency checks
   - Statistical analysis

## Logging Structure

Logs are organized by:
- Date and time
- Processing type
- Error details
- Performance metrics
- Analysis results

## Output Organization

Results are organized in:
- `data/masks/`: Generated masks
- `data/masked_images/`: Images with applied masks
- `data/analysis/`: Analysis results and visualizations
- `logs/`: Processing logs
- `saved_images/`: Additional visualization results 