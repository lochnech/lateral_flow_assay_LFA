# UNET Implementation

This document describes the UNET architecture and implementation used for LFA image segmentation.

## Model Architecture

The UNET model is implemented in `src/UNET/segmentation_ROI.py` and consists of:

### Encoder (Downsampling Path)
- Input: RGB image (3 channels)
- Feature channels: [64, 128, 256, 512]
- Each level includes:
  - Double convolution block
  - Batch normalization
  - ReLU activation
  - Max pooling

### Decoder (Upsampling Path)
- Feature channels: [512, 256, 128, 64]
- Each level includes:
  - Transposed convolution
  - Skip connection concatenation
  - Double convolution block
  - Batch normalization
  - ReLU activation

### Final Layer
- Output: Single channel binary mask
- Uses 1x1 convolution

## Key Features

1. **Skip Connections**
   - Connects encoder and decoder paths
   - Preserves spatial information
   - Improves segmentation accuracy

2. **Batch Normalization**
   - Stabilizes training
   - Improves convergence
   - Regularizes the model

3. **Flexible Architecture**
   - Configurable feature channels
   - Adaptable to different input sizes
   - Maintains aspect ratio

## Usage

```python
from src.UNET.segmentation_ROI import UNET

# Create model
model = UNET(
    in_channels=3,      # RGB input
    out_channels=1,     # Binary mask
    features=[64, 128, 256, 512]  # Feature channels
)

# Move to device
model = model.to(device)

# Forward pass
predictions = model(input_tensor)
```

## Model Output

The model outputs a single-channel tensor where:
- Values range from 0 to 1 (after sigmoid)
- Threshold at 0.5 for binary mask
- 1 represents the region of interest
- 0 represents the background

## Training Considerations

1. **Loss Function**
   - Uses BCEWithLogitsLoss
   - Handles class imbalance
   - Stable training

2. **Optimization**
   - Adam optimizer
   - Learning rate: 1e-4
   - Weight decay: 1e-6

3. **Data Augmentation**
   - Random rotation
   - Horizontal/vertical flips
   - Normalization

## Performance

The model is designed to:
- Process 512x512 images efficiently
- Maintain high accuracy on LFA images
- Handle varying background conditions
- Provide consistent segmentation results 