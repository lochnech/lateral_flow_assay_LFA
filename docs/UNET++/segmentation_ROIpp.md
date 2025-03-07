# segmentation_ROIpp

## Overview
This module implements a UNET++ architecture for image segmentation, which is an enhanced version of the traditional U-NET. UNET++ features nested and dense skip connections, providing better feature reuse and gradient flow.

## Architecture Details
The UNET++ implementation consists of:
- Nested encoder-decoder structure
- Dense skip connections between nodes
- Multiple deep supervision outputs
- Feature reuse through dense connections

## Classes

### UNETpp
The main UNET++ model class that implements the complete architecture.

#### Parameters
- `in_channels` (int): Number of input channels (default: 3 for RGB)
- `out_channels` (int): Number of output channels (default: 1 for binary mask)

#### Methods
- `__init__`: Initializes the UNET++ model with specified channels
- `forward`: Performs forward pass through the network
  - Input: Tensor of shape (batch_size, in_channels, height, width)
  - Output: List of tensors for deep supervision outputs

### VGGBlock
A VGG-style block used in the UNET++ architecture.

#### Parameters
- `in_channels` (int): Number of input channels
- `out_channels` (int): Number of output channels

#### Methods
- `__init__`: Initializes the VGG block
- `forward`: Performs forward pass through the block
  - Applies two 3x3 convolutions with batch normalization and ReLU

## Usage Example
```python
from segmentation_ROIpp import UNETpp

# Initialize model
model = UNETpp(in_channels=3, out_channels=1)

# Forward pass (returns multiple outputs for deep supervision)
outputs = model(input_tensor)  # input_tensor shape: (batch_size, 3, height, width)
```

## Technical Details
- Uses He initialization for weights
- Implements batch normalization for stable training
- Uses ReLU activation functions
- Supports deep supervision training
- Maintains spatial dimensions through padding