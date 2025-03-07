# segmentation_ROI

## Overview
This module implements a U-NET architecture for image segmentation, specifically designed for detecting regions of interest in Lateral Flow Assay (LFA) images. The U-NET architecture is particularly effective for biomedical image segmentation due to its encoder-decoder structure and skip connections.

## Architecture Details
The U-NET implementation consists of:
- Encoder path: Downsamples the input image while increasing feature channels
- Decoder path: Upsamples the feature maps while decreasing channels
- Skip connections: Connects corresponding encoder and decoder layers
- Final output: Single channel binary mask

## Classes

### UNET
The main U-NET model class that implements the complete architecture.

#### Parameters
- `in_channels` (int): Number of input channels (default: 3 for RGB)
- `out_channels` (int): Number of output channels (default: 1 for binary mask)

#### Methods
- `__init__`: Initializes the U-NET model with specified channels
- `forward`: Performs forward pass through the network
  - Input: Tensor of shape (batch_size, in_channels, height, width)
  - Output: Tensor of shape (batch_size, out_channels, height, width)

### DoubleConv
A double convolution block used throughout the U-NET architecture.

#### Parameters
- `in_channels` (int): Number of input channels
- `out_channels` (int): Number of output channels

#### Methods
- `__init__`: Initializes the double convolution block
- `forward`: Performs forward pass through the block
  - Applies two 3x3 convolutions with batch normalization and ReLU

## Usage Example
```python
from segmentation_ROI import UNET

# Initialize model
model = UNET(in_channels=3, out_channels=1)

# Forward pass
output = model(input_tensor)  # input_tensor shape: (batch_size, 3, height, width)
```

## Technical Details
- Uses He initialization for weights
- Implements batch normalization for stable training
- Uses ReLU activation functions
- Maintains spatial dimensions through padding