# API Documentation

## Core Modules

### Mask Generation
```python
from lfa.core.mask_generation import (
    setup_logging,
    get_background_color,
    create_smooth_transition,
    enhance_contrast,
    generate_transformed_data,
    generate_mask
)
```

#### Functions
- `setup_logging(log_dir: str, name: str = 'mask_generation') -> logging.Logger`
  - Sets up logging for mask generation
  - Returns configured logger instance

- `get_background_color(image: np.ndarray) -> Tuple[int, int, int]`
  - Detects the background color of an image
  - Returns RGB tuple

- `create_smooth_transition(image: np.ndarray, transition_width: int = 10) -> np.ndarray`
  - Creates smooth transitions at image edges
  - Returns processed image

- `enhance_contrast(image: np.ndarray) -> np.ndarray`
  - Enhances image contrast
  - Returns enhanced image

- `generate_transformed_data(image: np.ndarray) -> np.ndarray`
  - Generates transformed data for mask generation
  - Returns transformed image

- `generate_mask(image: np.ndarray, model: torch.nn.Module) -> np.ndarray`
  - Generates mask for input image using model
  - Returns binary mask

### Mask Application
```python
from lfa.core.mask_application import apply_mask_to_image
```

#### Functions
- `apply_mask_to_image(image: np.ndarray, mask: np.ndarray, output_path: Optional[str] = None) -> np.ndarray`
  - Applies mask to image
  - Returns masked image

### Analysis
```python
from lfa.core.analysis import (
    analyze_mask,
    compare_masks,
    generate_plots
)
```

#### Functions
- `analyze_mask(mask: np.ndarray) -> Dict[str, float]`
  - Analyzes mask properties
  - Returns dictionary of statistics

- `compare_masks(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> Dict[str, float]`
  - Compares predicted and ground truth masks
  - Returns dictionary of metrics

- `generate_plots(stats_df: pd.DataFrame, output_dir: Path) -> None`
  - Generates analysis plots
  - Saves plots to output directory

## Utility Modules

### Data Utilities
```python
from lfa.utils.data_utils import (
    create_data_loaders,
    get_transforms,
    LFAImageDataset
)
```

#### Functions
- `create_data_loaders(image_dir: str, mask_dir: str, batch_size: int = 8, val_split: float = 0.2, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]`
  - Creates training and validation data loaders
  - Returns tuple of data loaders

- `get_transforms(train: bool = True) -> albumentations.Compose`
  - Gets data augmentation transforms
  - Returns transform composition

#### Classes
- `LFAImageDataset(image_dir: str, mask_dir: str, transform: Optional[Callable] = None)`
  - Dataset class for LFA images and masks
  - Inherits from torch.utils.data.Dataset

### Image Utilities
```python
from lfa.utils.image_utils import (
    load_image,
    save_image,
    overlay_mask,
    calculate_iou,
    preprocess_image,
    postprocess_mask
)
```

#### Functions
- `load_image(image_path: str) -> Optional[np.ndarray]`
  - Loads image from path
  - Returns image array or None if loading fails

- `save_image(image: np.ndarray, output_path: str) -> bool`
  - Saves image to path
  - Returns success status

- `overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.5) -> np.ndarray`
  - Overlays mask on image
  - Returns overlaid image

- `calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float`
  - Calculates Intersection over Union
  - Returns IoU score

- `preprocess_image(image: np.ndarray) -> np.ndarray`
  - Preprocesses image for model input
  - Returns processed image

- `postprocess_mask(mask: np.ndarray) -> np.ndarray`
  - Postprocesses model output mask
  - Returns binary mask

### Logging Utilities
```python
from lfa.utils.logging_utils import setup_logging
```

#### Functions
- `setup_logging(log_dir: str, name: str = 'default', level: int = logging.INFO, maxBytes: int = 10485760, backupCount: int = 5) -> logging.Logger`
  - Sets up logging configuration
  - Returns configured logger

## Models

### UNet
```python
from lfa.models.unet import UNet
```

#### Classes
- `UNet(n_channels: int = 3, n_classes: int = 1, bilinear: bool = False)`
  - UNet model for image segmentation
  - Inherits from torch.nn.Module

#### Components
- `DoubleConv(in_channels: int, out_channels: int)`
  - Double convolution block
  - Used in UNet architecture

- `Down(in_channels: int, out_channels: int)`
  - Downsampling block
  - Used in UNet encoder

- `Up(in_channels: int, out_channels: int, bilinear: bool = True)`
  - Upsampling block
  - Used in UNet decoder

- `OutConv(in_channels: int, out_channels: int)`
  - Output convolution block
  - Used in UNet final layer

## Scripts

### Training Script
```bash
python scripts/train.py [--config CONFIG_PATH]
```

### Mask Generation Script
```bash
python scripts/generate_masks.py [--config CONFIG_PATH]
```

### Mask Application Script
```bash
python scripts/apply_masks.py [--config CONFIG_PATH]
```

### Analysis Script
```bash
python scripts/analyze_results.py [--config CONFIG_PATH]
```

### Test Runner Script
```bash
python scripts/run_tests.py [test_paths...] [-v] [-c]
``` 