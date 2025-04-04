import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LFADataset(Dataset):
    """Dataset for LFA image segmentation"""
    def __init__(self, 
                 image_dir: str, 
                 mask_dir: Optional[str] = None,
                 transform: Optional[A.Compose] = None,
                 image_size: Tuple[int, int] = (256, 256)):
        """
        Args:
            image_dir (str): Directory with all the images
            mask_dir (str, optional): Directory with all the masks
            transform (albumentations.Compose, optional): Optional transform to be applied
            image_size (tuple): Size to resize images to
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.image_size = image_size
        
        # Get list of image files
        self.image_files = sorted([f for f in self.image_dir.glob('*.png')])
        
        # If mask directory is provided, verify that each image has a corresponding mask
        if self.mask_dir:
            self.mask_files = []
            for img_path in self.image_files:
                mask_path = self.mask_dir / img_path.name
                if not mask_path.exists():
                    raise FileNotFoundError(f"Mask not found for image {img_path}")
                self.mask_files.append(mask_path)
        else:
            self.mask_files = None

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        
        # Load mask if available
        if self.mask_files:
            mask = cv2.imread(str(self.mask_files[idx]), 0)
            mask = cv2.resize(mask, self.image_size)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = None
        
        # Apply transforms
        if self.transform:
            if mask is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            else:
                transformed = self.transform(image=image)
                image = transformed['image']
        
        return image, mask

def get_train_transform(image_size: Tuple[int, int] = (256, 256)) -> A.Compose:
    """Get training data augmentation transforms
    
    Args:
        image_size (tuple): Size to resize images to
        
    Returns:
        albumentations.Compose: Training transforms
    """
    return A.Compose([
        A.Resize(*image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transform(image_size: Tuple[int, int] = (256, 256)) -> A.Compose:
    """Get validation data transforms
    
    Args:
        image_size (tuple): Size to resize images to
        
    Returns:
        albumentations.Compose: Validation transforms
    """
    return A.Compose([
        A.Resize(*image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def create_data_loaders(
    train_image_dir: str,
    train_mask_dir: str,
    val_image_dir: str,
    val_mask_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (256, 256)
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders
    
    Args:
        train_image_dir (str): Directory with training images
        train_mask_dir (str): Directory with training masks
        val_image_dir (str): Directory with validation images
        val_mask_dir (str): Directory with validation masks
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        image_size (tuple): Size to resize images to
        
    Returns:
        tuple: Training and validation data loaders
    """
    # Create datasets
    train_dataset = LFADataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        transform=get_train_transform(image_size),
        image_size=image_size
    )
    
    val_dataset = LFADataset(
        image_dir=val_image_dir,
        mask_dir=val_mask_dir,
        transform=get_val_transform(image_size),
        image_size=image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
