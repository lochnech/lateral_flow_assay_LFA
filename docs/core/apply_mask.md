# apply_mask

## Overview
This module provides functionality for applying binary masks to images, including operations like masking, cropping, and visualization. It's particularly useful for processing LFA images with their corresponding segmentation masks.

## Functions

### apply_mask
Applies a binary mask to an image.

#### Parameters
- `image` (np.ndarray): Input image
- `mask` (np.ndarray): Binary mask
- `background_color` (tuple): Color for masked regions (default: black)

#### Returns
- `masked_image` (np.ndarray): Image with mask applied

### crop_to_mask
Crops an image to the bounding box of the mask.

#### Parameters
- `image` (np.ndarray): Input image
- `mask` (np.ndarray): Binary mask
- `padding` (int): Padding around the mask (default: 0)

#### Returns
- `cropped_image` (np.ndarray): Cropped image
- `cropped_mask` (np.ndarray): Cropped mask

## Usage Example
```python
from apply_mask import apply_mask, crop_to_mask

# Load image and mask
image = cv2.imread("image.jpg")
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

# Apply mask
masked = apply_mask(image, mask)

# Crop to mask
cropped_img, cropped_mask = crop_to_mask(image, mask)
```

## Technical Details
- Supports different background colors
- Handles various image formats
- Implements efficient cropping algorithms
- Maintains mask alignment