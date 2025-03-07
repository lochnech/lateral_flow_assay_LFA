# dataset

## Overview
This module provides dataset functionality for loading and preprocessing image-mask pairs for the LFA segmentation model. It implements a custom PyTorch Dataset class that handles image loading, mask processing, and data augmentation.

## Classes

### LFADataset
A PyTorch Dataset class specifically designed for LFA image-mask pairs.

#### Parameters
- `image_dir` (str): Directory containing input images
- `mask_dir` (str): Directory containing corresponding mask images
- `transform` (callable, optional): Transform to apply to both image and mask

#### Methods
- `__init__`: Initializes the dataset
  - Sets up image and mask directories
  - Stores list of available images
  - Configures optional transforms

- `__len__`: Returns the number of samples in the dataset
  - Returns: int, number of image-mask pairs

- `__getitem__`: Loads and returns an image-mask pair
  - Input: index (int) - Index of the sample to load
  - Returns: tuple (image, mask)
    - image: Tensor of shape (channels, height, width)
    - mask: Tensor of shape (1, height, width)

## Data Format Requirements
- Images: RGB format (3 channels)
- Masks: Binary format (0 and 1)
- Mask files should be named with "_mask.gif" suffix
- Images and masks should have matching dimensions

## Usage Example
```python
from dataset import LFADataset
import albumentations as A

# Define transforms
transform = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ToTensorV2(),
])

# Create dataset
dataset = LFADataset(
    image_dir="./data/train_images/",
    mask_dir="./data/train_masks/",
    transform=transform
)

# Get a sample
image, mask = dataset[0]
```

## Technical Details
- Uses PIL for image loading
- Implements proper error handling
- Supports data augmentation through transforms
- Maintains image-mask alignment