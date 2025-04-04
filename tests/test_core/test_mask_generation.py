import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path
import cv2

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from lfa.core.mask_generation import (
    setup_logging,
    get_background_color,
    create_smooth_transition,
    enhance_contrast,
    generate_transformed_data,
    generate_mask
)

@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple test image with a white square on black background
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[64:192, 64:192] = 255
    return img

@pytest.fixture
def sample_mask():
    """Create a sample test mask"""
    # Create a binary mask matching the sample image
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[64:192, 64:192] = 255
    return mask

@pytest.fixture
def model():
    """Create a simple UNet model for testing"""
    from lfa.models.unet import UNet
    model = UNet(n_channels=3, n_classes=1)
    return model

def test_get_background_color(sample_image):
    """Test background color detection"""
    bg_color = get_background_color(sample_image)
    assert isinstance(bg_color, tuple)
    assert len(bg_color) == 3
    assert all(0 <= c <= 255 for c in bg_color)
    # For our test image, background should be black
    assert bg_color == (0, 0, 0)

def test_create_smooth_transition(sample_image):
    """Test smooth transition creation"""
    transition = create_smooth_transition(sample_image, 10)
    assert isinstance(transition, np.ndarray)
    assert transition.shape == sample_image.shape
    assert transition.dtype == np.uint8

def test_enhance_contrast(sample_image):
    """Test contrast enhancement"""
    enhanced = enhance_contrast(sample_image)
    assert isinstance(enhanced, np.ndarray)
    assert enhanced.shape == sample_image.shape
    assert enhanced.dtype == np.uint8
    # Enhanced image should have higher contrast
    assert np.std(enhanced) >= np.std(sample_image)

def test_generate_transformed_data(sample_image):
    """Test transformed data generation"""
    transformed = generate_transformed_data(sample_image)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == sample_image.shape
    assert transformed.dtype == np.uint8

def test_generate_mask(sample_image, model):
    """Test mask generation"""
    # Set model to evaluation mode
    model.eval()
    
    # Generate mask
    mask = generate_mask(sample_image, model)
    
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (sample_image.shape[0], sample_image.shape[1])
    assert mask.dtype == np.uint8
    # Mask should be binary
    assert np.all(np.logical_or(mask == 0, mask == 255))

def test_mask_generation_pipeline(tmp_path, sample_image, model):
    """Test the complete mask generation pipeline"""
    # Create temporary directories
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    # Save sample image
    image_path = input_dir / "test.png"
    cv2.imwrite(str(image_path), sample_image)
    
    # Setup logging
    logger = setup_logging(str(tmp_path / "logs"))
    
    # Generate masks for all images
    for img_path in input_dir.glob("*.png"):
        mask = generate_mask(cv2.imread(str(img_path)), model)
        output_path = output_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(output_path), mask)
    
    # Verify output
    assert len(list(output_dir.glob("*.png"))) == 1
    output_mask = cv2.imread(str(output_dir / "test_mask.png"), 0)
    assert output_mask is not None
    assert output_mask.shape == sample_image.shape[:2]
