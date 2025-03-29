# Core Functionality

This document describes the core functionality of the LFA analysis pipeline.

## Image Processing Pipeline

The core functionality is implemented in `src/core/` and includes:

### 1. Mask Generation (`generate_with_UNET.py`)

Main features:
- Background color detection
- Image padding
- Mask generation
- Logging and visualization

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
- Mask overlay
- ROI extraction
- Result visualization
- Batch processing

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

### Visualization
- Original vs transformed images
- Mask overlays
- Processing results
- Error cases

## Usage Examples

### Generate Masks
```python
from src.core.generate_with_UNET import generate_mask

# Generate mask for single image
generate_mask(
    image_path="data/test_images/image.png",
    save_path="data/result_images/mask.png"
)
```

### Apply Masks
```python
from src.core.apply_mask import apply_mask_to_image

# Apply mask to image
apply_mask_to_image(
    image_path="data/test_images/image.png",
    mask_path="data/result_images/mask.png",
    output_path="data/result_images/result.png"
)
```

## Error Handling

The pipeline includes robust error handling:
1. File existence checks
2. Image loading validation
3. Processing error recovery
4. Detailed error logging

## Performance Considerations

1. **Memory Management**
   - GPU memory optimization
   - Batch processing
   - Resource cleanup

2. **Processing Speed**
   - Efficient image operations
   - Parallel processing where possible
   - Optimized data loading

3. **Quality Control**
   - Input validation
   - Output verification
   - Consistency checks

## Logging Structure

Logs are organized by:
- Date and time
- Processing type
- Error details
- Performance metrics

## Output Organization

Results are organized in:
- `data/result_images/`: Generated masks
- `data/padded_images/`: Preprocessed images
- `logs/`: Processing logs
- `saved_images/`: Visualization results 