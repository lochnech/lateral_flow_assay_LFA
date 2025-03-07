# main

## Overview
This module implements traditional computer vision techniques for detecting red lines in LFA images. It uses color thresholding, edge detection, and Hough transform to identify regions of interest without using deep learning.

## Functions

### process_image
Processes an image to detect red lines using color-based segmentation and line detection.

#### Parameters
- `image_path` (str): Path to input image
- `output_path` (str): Path to save processed image

#### Process Flow
1. Image Loading and Preprocessing
   - Loads image using OpenCV
   - Converts to HSV color space for better color segmentation

2. Red Region Detection
   - Creates masks for red regions using HSV thresholds
   - Applies morphological operations to clean up the mask

3. Line Detection
   - Applies Canny edge detection
   - Uses Hough transform to detect lines
   - Filters lines based on length and angle

4. Visualization
   - Draws detected lines on original image
   - Saves result to specified output path

#### Usage Example
```python
from main import process_image

# Process a single image
process_image("input.jpg", "output.jpg")
```

## Technical Details
- Uses HSV color space for robust color detection
- Implements adaptive thresholding
- Applies morphological operations for noise removal
- Uses probabilistic Hough transform for line detection
- Supports both horizontal and vertical line detection