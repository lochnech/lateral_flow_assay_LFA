import logging
import os
from pathlib import Path
from typing import Optional
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime

def setup_logging(log_dir: str, name: str = 'lfa') -> logging.Logger:
    """Set up logging configuration
    
    Args:
        log_dir (str): Directory to save log files
        name (str): Name of the logger
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create log directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_dir / f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

class TensorBoardLogger:
    """TensorBoard logger for training metrics"""
    def __init__(self, log_dir: str):
        """Initialize TensorBoard logger
        
        Args:
            log_dir (str): Directory to save TensorBoard logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        self.step = 0
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log a scalar value
        
        Args:
            tag (str): Tag for the scalar
            value (float): Value to log
            step (int, optional): Step number
        """
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)
    
    def log_image(self, tag: str, image: np.ndarray, step: Optional[int] = None):
        """Log an image
        
        Args:
            tag (str): Tag for the image
            image (np.ndarray): Image to log
            step (int, optional): Step number
        """
        if step is None:
            step = self.step
        self.writer.add_image(tag, image, step, dataformats='HWC')
    
    def log_histogram(self, tag: str, values: np.ndarray, step: Optional[int] = None):
        """Log a histogram
        
        Args:
            tag (str): Tag for the histogram
            values (np.ndarray): Values to log
            step (int, optional): Step number
        """
        if step is None:
            step = self.step
        self.writer.add_histogram(tag, values, step)
    
    def log_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor):
        """Log model graph
        
        Args:
            model (torch.nn.Module): Model to log
            input_tensor (torch.Tensor): Example input tensor
        """
        self.writer.add_graph(model, input_tensor)
    
    def increment_step(self):
        """Increment the step counter"""
        self.step += 1
    
    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()

class TrainingMonitor:
    """Monitor training progress"""
    def __init__(self, log_dir: str):
        """Initialize training monitor
        
        Args:
            log_dir (str): Directory to save logs
        """
        self.logger = setup_logging(log_dir)
        self.tb_logger = TensorBoardLogger(log_dir)
        self.best_loss = float('inf')
        self.best_epoch = 0
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                 train_metrics: Optional[dict] = None, val_metrics: Optional[dict] = None):
        """Log epoch results
        
        Args:
            epoch (int): Current epoch
            train_loss (float): Training loss
            val_loss (float): Validation loss
            train_metrics (dict, optional): Training metrics
            val_metrics (dict, optional): Validation metrics
        """
        # Log to console and file
        self.logger.info(f'Epoch {epoch}:')
        self.logger.info(f'  Train Loss: {train_loss:.4f}')
        self.logger.info(f'  Val Loss: {val_loss:.4f}')
        
        if train_metrics:
            for name, value in train_metrics.items():
                self.logger.info(f'  Train {name}: {value:.4f}')
        
        if val_metrics:
            for name, value in val_metrics.items():
                self.logger.info(f'  Val {name}: {value:.4f}')
        
        # Log to TensorBoard
        self.tb_logger.log_scalar('Loss/train', train_loss, epoch)
        self.tb_logger.log_scalar('Loss/val', val_loss, epoch)
        
        if train_metrics:
            for name, value in train_metrics.items():
                self.tb_logger.log_scalar(f'Metrics/train_{name}', value, epoch)
        
        if val_metrics:
            for name, value in val_metrics.items():
                self.tb_logger.log_scalar(f'Metrics/val_{name}', value, epoch)
        
        # Update best model
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.logger.info(f'New best model at epoch {epoch} with val loss {val_loss:.4f}')
    
    def log_batch(self, batch_idx: int, loss: float, metrics: Optional[dict] = None):
        """Log batch results
        
        Args:
            batch_idx (int): Current batch index
            loss (float): Batch loss
            metrics (dict, optional): Batch metrics
        """
        self.tb_logger.log_scalar('Loss/batch', loss, self.tb_logger.step)
        
        if metrics:
            for name, value in metrics.items():
                self.tb_logger.log_scalar(f'Metrics/batch_{name}', value, self.tb_logger.step)
        
        self.tb_logger.increment_step()
    
    def close(self):
        """Close all loggers"""
        self.tb_logger.close()
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
