# pp_training

## Overview
This module implements the training pipeline for the UNET++ segmentation model. It includes the training loop, deep supervision loss calculation, optimization, and model checkpointing.

## Functions

### train_fn
Performs one epoch of training with deep supervision.

#### Parameters
- `loader` (DataLoader): Training data loader
- `model` (nn.Module): UNET++ model to train
- `optimizer` (torch.optim.Optimizer): Optimizer for training
- `loss_fn` (callable): Loss function
- `epoch` (int): Current epoch number

#### Returns
- `avg_loss` (float): Average training loss
- `deep_losses` (list): List of losses for each deep supervision output

### main
Main training loop implementation for UNET++.

#### Configuration
- Learning rate: 1e-4
- Batch size: 4
- Number of epochs: 100
- Image dimensions: 512x512
- Device: CUDA if available, else CPU

#### Training Process
1. Sets up data transforms
2. Initializes UNET++ model, loss function, and optimizer
3. Creates data loaders
4. Runs training loop with:
   - Forward pass with deep supervision
   - Weighted loss calculation
   - Backward pass
   - Optimization
   - Checkpointing
   - Validation

## Usage Example
```python
if __name__ == "__main__":
    main()
```

## Technical Details
- Implements deep supervision training
- Uses weighted loss combination
- Supports mixed precision training
- Implements learning rate scheduling
- Saves training metrics
- Supports model checkpointing