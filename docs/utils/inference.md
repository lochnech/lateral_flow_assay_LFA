# inference

## Overview
This module handles model inference on new images, providing functions for both single image and batch processing. It includes visualization tools and utilities for saving and displaying predictions.

## Functions

### pred_show_image_grid
Shows predictions for multiple images in a grid layout.

#### Parameters
- `loader` (DataLoader): Data loader containing images
- `model` (nn.Module): Trained model for inference
- `device` (str): Device to run inference on
- `folder` (str): Folder to save results
- `batch_size` (int): Number of images to process at once

#### Returns
- Saves grid visualization of predictions
- Displays prediction results

### single_image_inference
Performs inference on a single image using the trained model.

#### Parameters
- `image_path` (str): Path to input image
- `model_path` (str): Path to trained model
- `device` (str): Device to run inference on

#### Returns
- `mask` (np.ndarray): Predicted binary mask

## Usage Example
```python
from inference import single_image_inference

# Run inference on a single image
mask = single_image_inference("input.jpg", "model.pth.tar")
```

## Technical Details
- Supports both CPU and GPU inference
- Implements proper image preprocessing
- Generates visualizations of predictions
- Handles batch processing efficiently