# Lateral Flow Assay (LFA) Analysis Documentation

This documentation provides a comprehensive overview of the LFA analysis project, which uses deep learning (UNET) for segmentation and analysis of lateral flow assay images.

## Project Structure

```
lateral_flow_assay_LFA/
├── data/                    # Data directory
│   ├── test_images/        # Test images
│   ├── train_images/       # Training images
│   ├── train_masks/        # Training masks
│   ├── masks/             # Generated masks
│   ├── masked_images/     # Images with applied masks
│   └── analysis/          # Analysis results
├── src/                    # Source code
│   ├── core/              # Core functionality
│   │   ├── generate_with_UNET.py  # Mask generation
│   │   └── apply_mask.py          # Mask application
│   ├── UNET/              # UNET implementation
│   └── utils/             # Utility functions
├── models/                # Saved model checkpoints
├── docs/                  # Documentation
├── logs/                  # Training and processing logs
└── saved_images/         # Generated output images
```

## Key Components

1. [UNET Implementation](UNET.md)
   - Model architecture
   - Training process
   - Segmentation approach

2. [Core Functionality](core.md)
   - Image processing pipeline
   - Mask generation and application
   - Background color detection
   - Analysis and visualization

3. [Data Management](data.md)
   - Data organization
   - Image naming conventions
   - Data preprocessing

4. [Training Process](training.md)
   - Training configuration
   - Loss functions
   - Checkpoint management

5. [Usage Guide](usage.md)
   - Installation
   - Running the pipeline
   - Common use cases

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python src/UNET/training.py
```

3. Generate masks:
```bash
python src/core/generate_with_UNET.py
```

4. Apply masks to images:
```bash
python src/core/apply_mask.py
```

5. Analyze results:
```bash
python src/utils/analyze_masks.py
```

## Requirements

- Python 3.8+
- PyTorch
- Albumentations
- OpenCV
- NumPy
- Pillow
- Matplotlib

See `requirements.txt` for full dependencies. 