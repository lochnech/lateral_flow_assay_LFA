# Configuration Guide

## Configuration Files

The project uses YAML configuration files to manage various settings. The main configuration files are:

1. `lfa/config/default_config.yaml` - Default configuration for all operations
2. `lfa/config/training_config.yaml` - Training-specific configuration
3. `lfa/config/mask_generation_config.yaml` - Mask generation configuration
4. `lfa/config/mask_application_config.yaml` - Mask application configuration
5. `lfa/config/analysis_config.yaml` - Analysis configuration

## Configuration Structure

### Default Configuration
```yaml
# Default configuration for all operations
model:
  type: "unet"  # or "unet++"
  n_channels: 3
  n_classes: 1
  bilinear: false

data:
  image_size: [256, 256]
  batch_size: 8
  num_workers: 4
  val_split: 0.2

paths:
  data_dir: "data"
  output_dir: "output"
  model_dir: "models"
  log_dir: "logs"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Training Configuration
```yaml
# Training-specific configuration
training:
  epochs: 100
  early_stopping:
    patience: 10
    min_delta: 0.001
  checkpoint:
    save_freq: 5
    save_best_only: true

optimizer:
  type: "adam"  # or "sgd"
  learning_rate: 0.001
  weight_decay: 0.0001
  momentum: 0.9  # for SGD

scheduler:
  type: "reduce_lr"  # or "cosine"
  patience: 5
  factor: 0.1
  min_lr: 1e-6

data_augmentation:
  train:
    - type: "horizontal_flip"
      p: 0.5
    - type: "vertical_flip"
      p: 0.5
    - type: "rotate"
      limit: 30
      p: 0.5
    - type: "brightness_contrast"
      brightness_limit: 0.2
      contrast_limit: 0.2
      p: 0.5
```

### Mask Generation Configuration
```yaml
# Mask generation configuration
mask_generation:
  model_checkpoint: "models/best_model.pth"
  device: "cuda"  # or "cpu"
  threshold: 0.5
  postprocess:
    min_area: 100
    smoothing: true
    kernel_size: 5

image_processing:
  contrast_enhancement: true
  smooth_transition: true
  transition_width: 10
```

### Mask Application Configuration
```yaml
# Mask application configuration
mask_application:
  output_format: "png"
  overlay:
    enabled: true
    color: [0, 255, 0]  # RGB
    alpha: 0.5
  save_original: true
```

### Analysis Configuration
```yaml
# Analysis configuration
analysis:
  metrics:
    - "iou"
    - "dice"
    - "precision"
    - "recall"
    - "f1"
  plots:
    - type: "histogram"
      metric: "iou"
    - type: "scatter"
      x: "area"
      y: "iou"
    - type: "box"
      metric: "dice"
  output:
    format: "png"
    dpi: 300
```

## Using Configuration Files

### Command Line Usage
All scripts accept a `--config` argument to specify a configuration file:

```bash
# Training
python scripts/train.py --config config/training_config.yaml

# Mask Generation
python scripts/generate_masks.py --config config/mask_generation_config.yaml

# Mask Application
python scripts/apply_masks.py --config config/mask_application_config.yaml

# Analysis
python scripts/analyze_results.py --config config/analysis_config.yaml
```

### Programmatic Usage
Configuration can be loaded in Python code:

```python
from lfa.utils.config_utils import load_config

# Load configuration
config = load_config("config/default_config.yaml")

# Access configuration values
model_type = config["model"]["type"]
learning_rate = config["optimizer"]["learning_rate"]
```

## Configuration Overrides

Configuration values can be overridden at runtime using environment variables. The environment variable name should be in the format:

```
LFA_CONFIG_SECTION_SUBSECTION_KEY=value
```

For example:
```bash
# Override learning rate
export LFA_CONFIG_OPTIMIZER_LEARNING_RATE=0.0001

# Override batch size
export LFA_CONFIG_DATA_BATCH_SIZE=16
```

## Best Practices

1. Always keep a backup of your default configuration files
2. Use version control for configuration files
3. Document any changes to configuration files
4. Test configuration changes in a development environment first
5. Use environment variables for sensitive information (API keys, etc.) 