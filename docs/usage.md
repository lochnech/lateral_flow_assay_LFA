# Usage Guide

This document provides comprehensive instructions for using the LFA analysis pipeline.

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd lateral_flow_assay_LFA
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Directory Setup

### Required Directories
```bash
mkdir -p data/{test_images,train_images,train_masks}
mkdir -p models
mkdir -p logs
mkdir -p saved_images
```

### Data Organization
1. Place training images in `data/train_images/`
2. Place training masks in `data/train_masks/`
3. Place test images in `data/test_images/`

## Training the Model

### Basic Training
```bash
python src/UNET/training.py
```

### Training Options
```bash
# Resume from checkpoint
python src/UNET/training.py --checkpoint models/model_checkpoint.pth.tar

# Start fresh (ignore existing checkpoints)
python src/UNET/training.py --reset
```

### Training Output
- Model checkpoints in `models/`
- Training logs in `logs/`
- Validation results in `saved_images/`

## Generating Masks

### Single Image
```python
from src.core.generate_with_UNET import generate_mask

generate_mask(
    image_path="data/test_images/image.png",
    save_path="data/result_images/mask.png"
)
```

### Batch Processing
```bash
python src/core/generate_with_UNET.py
```

### Output
- Generated masks in `data/result_images/`
- Processing logs in `logs/`
- Visualization in `saved_images/`

## Applying Masks

### Single Image
```python
from src.core.apply_mask import apply_mask_to_image

apply_mask_to_image(
    image_path="data/test_images/image.png",
    mask_path="data/result_images/mask.png",
    output_path="data/result_images/result.png"
)
```

### Batch Processing
```bash
python src/core/apply_mask.py
```

## Common Use Cases

### 1. Full Pipeline
```bash
# 1. Train model
python src/UNET/training.py

# 2. Generate masks
python src/core/generate_with_UNET.py

# 3. Apply masks
python src/core/apply_mask.py
```

### 2. Quick Testing
```bash
# Generate and apply mask for single image
python src/core/generate_with_UNET.py --image path/to/image.png
python src/core/apply_mask.py --image path/to/image.png --mask path/to/mask.png
```

### 3. Model Evaluation
```bash
# Train with validation
python src/UNET/training.py --validate

# Generate predictions
python src/core/generate_with_UNET.py --predict
```

## Configuration

### Model Parameters
- Image size: 512x512
- Batch size: 8
- Learning rate: 1e-4
- Epochs: 2000

### Data Augmentation
- Rotation: ±35 degrees
- Horizontal flip: 50%
- Vertical flip: 10%
- Normalization: zero mean, unit std

## Monitoring and Debugging

### Logs
- Check `logs/` for:
  - Training progress
  - Processing errors
  - Performance metrics

### Visualization
- View results in `saved_images/`
- Compare original and processed
- Check mask quality

### Common Issues

1. **Memory Errors**
   ```bash
   # Reduce batch size
   export BATCH_SIZE=4
   python src/UNET/training.py
   ```

2. **CUDA Errors**
   ```bash
   # Clear GPU cache
   python -c "import torch; torch.cuda.empty_cache()"
   ```

3. **File Not Found**
   ```bash
   # Check directory structure
   ls -R data/
   ```

## Best Practices

### 1. Data Management
- Use consistent naming
- Regular backups
- Version control
- Clean organization

### 2. Training
- Monitor validation
- Regular checkpoints
- Log analysis
- Performance tracking

### 3. Processing
- Verify inputs
- Check outputs
- Monitor logs
- Regular validation

## Support

### Getting Help
1. Check logs
2. Review documentation
3. Check common issues
4. Contact maintainers

### Reporting Issues
1. Include logs
2. Provide steps to reproduce
3. Share error messages
4. Describe environment 