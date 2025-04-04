import os
import sys
import pytest
import numpy as np
import torch
from pathlib import Path
import cv2

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from lfa.utils.data_utils import (
    create_data_loaders,
    get_transforms,
    LFAImageDataset
)

@pytest.fixture
def sample_image():
    """Create a sample test image"""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[64:192, 64:192] = 255
    return img

@pytest.fixture
def sample_mask():
    """Create a sample test mask"""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[64:192, 64:192] = 255
    return mask

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory with sample data"""
    # Create directories
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    
    # Create sample images and masks
    for i in range(5):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[64:192, 64:192] = 255
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[64:192, 64:192] = 255
        
        cv2.imwrite(str(image_dir / f"image_{i}.png"), img)
        cv2.imwrite(str(mask_dir / f"image_{i}.png"), mask)
    
    return tmp_path

def test_get_transforms():
    """Test transform creation"""
    # Test training transforms
    train_transforms = get_transforms(train=True)
    assert train_transforms is not None
    
    # Test validation transforms
    val_transforms = get_transforms(train=False)
    assert val_transforms is not None
    
    # Verify that training transforms include augmentation
    assert any(isinstance(t, type) for t in train_transforms.transforms)
    
    # Verify that validation transforms don't include augmentation
    assert not any(isinstance(t, type) for t in val_transforms.transforms)

def test_lfa_image_dataset(temp_data_dir, sample_image, sample_mask):
    """Test LFAImageDataset class"""
    # Create dataset
    dataset = LFAImageDataset(
        image_dir=str(temp_data_dir / "images"),
        mask_dir=str(temp_data_dir / "masks"),
        transform=get_transforms(train=True)
    )
    
    # Test dataset length
    assert len(dataset) == 5
    
    # Test getting an item
    image, mask = dataset[0]
    
    # Check types
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    
    # Check shapes
    assert image.shape == (3, 256, 256)
    assert mask.shape == (1, 256, 256)
    
    # Check value ranges
    assert torch.all(image >= 0) and torch.all(image <= 1)
    assert torch.all(mask >= 0) and torch.all(mask <= 1)

def test_create_data_loaders(temp_data_dir):
    """Test data loader creation"""
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        image_dir=str(temp_data_dir / "images"),
        mask_dir=str(temp_data_dir / "masks"),
        batch_size=2,
        val_split=0.2,
        num_workers=0
    )
    
    # Check loader types
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    
    # Check batch sizes
    assert train_loader.batch_size == 2
    assert val_loader.batch_size == 2
    
    # Check dataset sizes
    assert len(train_loader.dataset) == 4  # 80% of 5
    assert len(val_loader.dataset) == 1    # 20% of 5
    
    # Test loading a batch
    for images, masks in train_loader:
        assert images.shape == (2, 3, 256, 256)
        assert masks.shape == (2, 1, 256, 256)
        break

def test_dataset_with_invalid_data(temp_data_dir):
    """Test dataset with invalid data"""
    # Add an invalid image
    invalid_img_path = temp_data_dir / "images" / "invalid.png"
    with open(invalid_img_path, 'w') as f:
        f.write("not an image")
    
    # Create dataset
    dataset = LFAImageDataset(
        image_dir=str(temp_data_dir / "images"),
        mask_dir=str(temp_data_dir / "masks"),
        transform=get_transforms(train=True)
    )
    
    # Test that invalid images are skipped
    assert len(dataset) == 5  # Should still be 5 valid images

def test_data_loaders_with_different_splits(temp_data_dir):
    """Test data loaders with different validation splits"""
    # Test with 0% validation split
    train_loader, val_loader = create_data_loaders(
        image_dir=str(temp_data_dir / "images"),
        mask_dir=str(temp_data_dir / "masks"),
        batch_size=2,
        val_split=0.0,
        num_workers=0
    )
    assert len(train_loader.dataset) == 5
    assert len(val_loader.dataset) == 0
    
    # Test with 100% validation split
    train_loader, val_loader = create_data_loaders(
        image_dir=str(temp_data_dir / "images"),
        mask_dir=str(temp_data_dir / "masks"),
        batch_size=2,
        val_split=1.0,
        num_workers=0
    )
    assert len(train_loader.dataset) == 0
    assert len(val_loader.dataset) == 5
