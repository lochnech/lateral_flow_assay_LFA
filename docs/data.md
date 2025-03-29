# Data Management

This document describes the data organization and management in the LFA analysis project.

## Directory Structure

```
data/
├── test_images/           # Test images
│   ├── 1 hour white 3/    # 1-hour time point, set 3
│   ├── 12 hours white 01/ # 12-hour time point, set 1
│   └── 24 hours white 01/ # 24-hour time point, set 1
├── train_images/          # Training images
└── train_masks/           # Training masks
```

## Image Naming Convention

Images follow a consistent naming scheme:
```
white_XX_YYhour_ZZZ.png
```
Where:
- `XX`: Set number (01, 02, etc.)
- `YY`: Time point (01, 12, 24)
- `ZZZ`: Sequential number (001-007)

Example:
- `white_01_12hour_001.png`: Set 1, 12-hour time point, first image
- `white_03_01hour_007.png`: Set 3, 1-hour time point, seventh image

## Data Organization

### Time Points
- 1 hour
- 12 hours
- 24 hours

### Sets
- White background sets (01, 02, 03)
- Each set contains multiple images

### Image Types
1. **Raw Images**
   - Original LFA images
   - RGB format
   - Various sizes

2. **Processed Images**
   - Resized to 512x512
   - Normalized
   - Padded with background color

3. **Masks**
   - Binary masks
   - Single channel
   - 512x512 resolution

## Data Preprocessing

### Image Processing
1. **Resizing**
   - Maintain aspect ratio
   - Scale to fit 512x512
   - Preserve image quality

2. **Padding**
   - Detect background color
   - Apply consistent padding
   - Center image content

3. **Normalization**
   - Zero mean
   - Unit standard deviation
   - Consistent value range

### Mask Generation
1. **Training Masks**
   - Manual annotation
   - Binary format
   - ROI marking

2. **Generated Masks**
   - Model predictions
   - Thresholded output
   - Quality verification

## Data Validation

### Input Validation
- File existence checks
- Format verification
- Size requirements
- Color space validation

### Output Validation
- Mask quality checks
- Size verification
- Value range validation
- Consistency checks

## Data Backup

### Checkpoints
- Model checkpoints
- Training progress
- Best models

### Logs
- Processing logs
- Error logs
- Performance metrics

## Usage Guidelines

### Adding New Data
1. Follow naming convention
2. Place in appropriate directory
3. Verify format and quality
4. Update documentation

### Processing Data
1. Use provided scripts
2. Check logs for errors
3. Verify outputs
4. Maintain organization

## Best Practices

1. **File Management**
   - Use consistent naming
   - Maintain directory structure
   - Regular backups
   - Version control

2. **Quality Control**
   - Verify image quality
   - Check mask accuracy
   - Monitor processing logs
   - Regular validation

3. **Organization**
   - Clear directory structure
   - Consistent naming
   - Proper documentation
   - Regular cleanup 