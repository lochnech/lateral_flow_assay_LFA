import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from typing import Dict, Any

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from lfa.models.unet import UNet
from lfa.models.unetpp import NestedUNet
from lfa.utils.data_utils import create_data_loaders
from lfa.utils.logging_utils import TrainingMonitor
from lfa.utils.image_utils import calculate_iou

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_model(config: Dict[str, Any]) -> nn.Module:
    """Create model based on configuration
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        nn.Module: Created model
    """
    model_config = config['model']
    
    if model_config['name'] == 'unet':
        model = UNet(
            n_channels=model_config['input_channels'],
            n_classes=model_config['num_classes'],
            bilinear=model_config['bilinear']
        )
    elif model_config['name'] == 'unetpp':
        model = NestedUNet(
            num_classes=model_config['num_classes'],
            input_channels=model_config['input_channels'],
            deep_supervision=model_config['deep_supervision']
        )
    else:
        raise ValueError(f"Unknown model name: {model_config['name']}")
    
    return model

def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer based on configuration
    
    Args:
        model (nn.Module): Model to optimize
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        optim.Optimizer: Created optimizer
    """
    training_config = config['training']
    
    if training_config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
    elif training_config['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=training_config['learning_rate'],
            momentum=0.9,
            weight_decay=training_config['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {training_config['optimizer']}")
    
    return optimizer

def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler based on configuration
    
    Args:
        optimizer (optim.Optimizer): Optimizer to schedule
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        optim.lr_scheduler._LRScheduler: Created scheduler
    """
    training_config = config['training']
    
    if training_config['scheduler'] == 'reduce_lr_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=training_config['factor'],
            patience=training_config['patience'],
            min_lr=training_config['min_lr']
        )
    elif training_config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config['num_epochs'],
            eta_min=training_config['min_lr']
        )
    else:
        raise ValueError(f"Unknown scheduler: {training_config['scheduler']}")
    
    return scheduler

def train_epoch(model: nn.Module, 
                train_loader: DataLoader, 
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                monitor: TrainingMonitor) -> float:
    """Train for one epoch
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        optimizer (optim.Optimizer): Optimizer
        criterion (nn.Module): Loss function
        device (torch.device): Device to train on
        monitor (TrainingMonitor): Training monitor
        
    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        
        # Log batch results
        if batch_idx % monitor.tb_logger.log_interval == 0:
            monitor.log_batch(batch_idx, loss.item())
    
    return total_loss / len(train_loader)

def validate_epoch(model: nn.Module, 
                  val_loader: DataLoader,
                  criterion: nn.Module,
                  device: torch.device) -> Tuple[float, Dict[str, float]]:
    """Validate for one epoch
    
    Args:
        model (nn.Module): Model to validate
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to validate on
        
    Returns:
        Tuple[float, Dict[str, float]]: Average validation loss and metrics
    """
    model.eval()
    total_loss = 0
    total_iou = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            total_loss += loss.item()
            
            # Calculate IoU
            preds = (torch.sigmoid(outputs) > 0.5).float()
            iou = calculate_iou(preds.cpu().numpy(), masks.cpu().numpy())
            total_iou += iou
    
    avg_loss = total_loss / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    
    metrics = {
        'iou': avg_iou
    }
    
    return avg_loss, metrics

def main():
    # Load configuration
    config = load_config('lfa/config/training_config.yaml')
    
    # Set random seed
    torch.manual_seed(config['experiment']['seed'])
    np.random.seed(config['experiment']['seed'])
    
    # Create output directories
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['data']['log_dir'], exist_ok=True)
    
    # Set device
    device = torch.device(config['hardware']['device'])
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_image_dir=config['data']['train_image_dir'],
        train_mask_dir=config['data']['train_mask_dir'],
        val_image_dir=config['data']['val_image_dir'],
        val_mask_dir=config['data']['val_mask_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers'],
        image_size=tuple(config['augmentation']['image_size'])
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Create training monitor
    monitor = TrainingMonitor(config['data']['log_dir'])
    
    # Training loop
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, monitor)
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Log epoch results
        monitor.log_epoch(epoch, train_loss, val_loss, val_metrics=val_metrics)
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }
            
            torch.save(checkpoint, os.path.join(config['logging']['checkpoint_dir'], 'best_model.pth'))
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if early_stopping_counter >= config['training']['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Close monitor
    monitor.close()

if __name__ == "__main__":
    main()
