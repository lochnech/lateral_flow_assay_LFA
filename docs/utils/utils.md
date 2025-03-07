# utils

## Overview
This module provides utility functions for model training, evaluation, and data management. It includes functions for checkpoint management, data loading, accuracy evaluation, and prediction visualization.

## Functions

### save_checkpoint
Saves model checkpoint to disk with training state.

#### Parameters
- `state` (dict): Dictionary containing model state, optimizer state, and epoch
- `filename` (str): Path where checkpoint will be saved

#### Usage
```python
checkpoint = {
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch
}
save_checkpoint(checkpoint, "model_checkpoint.pth.tar")
```

### load_checkpoint
Loads model checkpoint from disk.

#### Parameters
- `checkpoint` (dict): Dictionary containing checkpoint data
- `model` (nn.Module): Model to load state into
- `optimizer` (torch.optim.Optimizer): Optimizer to load state into

#### Usage
```python
checkpoint = torch.load("model_checkpoint.pth.tar")
load_checkpoint(checkpoint, model, optimizer)
```

### get_loaders
Creates data loaders for training and validation.

#### Parameters
- `batch_size` (int): Number of samples per batch
- `train_transform` (callable): Transform to apply to training data
- `val_transform` (callable): Transform to apply to validation data
- `num_workers` (int): Number of data loading workers
- `pin_memory` (bool): Whether to pin memory for faster data transfer

#### Returns
- `train_loader` (DataLoader): Training data loader
- `val_loader` (DataLoader): Validation data loader

### check_accuracy
Evaluates model accuracy on a data loader.

#### Parameters
- `loader` (DataLoader): Data loader to evaluate on
- `model` (nn.Module): Model to evaluate
- `device` (str): Device to run evaluation on

#### Returns
- `dice_score` (float): Dice score for segmentation accuracy

### save_predictions_as_image
Saves model predictions as images.

#### Parameters
- `loader` (DataLoader): Data loader containing images
- `model` (nn.Module): Model to generate predictions
- `folder` (str): Folder to save predictions
- `device` (str): Device to run predictions on

## Technical Details
- Implements proper error handling
- Supports GPU acceleration
- Uses efficient data loading with pin_memory
- Implements proper checkpoint management 