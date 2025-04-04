import os
import sys
import pytest
import numpy as np
from pathlib import Path
import cv2

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from lfa.utils.image_utils import (
    load_image,
    save_image,
    overlay_mask,
    calculate_iou,
    preprocess_image,
    postprocess_mask
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

def test_load_image(tmp_path, sample_image):
    """Test image loading"""
    # Save and load image
    img_path = tmp_path / "test.png"
    cv2.imwrite(str(img_path), sample_image)
    
    # Test loading existing image
    loaded_img = load_image(str(img_path))
    assert loaded_img is not None
    assert np.array_equal(loaded_img, sample_image)
    
    # Test loading non-existent image
    assert load_image(str(tmp_path / "nonexistent.png")) is None

def test_save_image(tmp_path, sample_image):
    """Test image saving"""
    # Save image
    img_path = tmp_path / "test.png"
    save_image(sample_image, str(img_path))
    
    # Verify image was saved
    assert img_path.exists()
    loaded_img = cv2.imread(str(img_path))
    assert loaded_img is not None
    assert np.array_equal(loaded_img, sample_image)

def test_overlay_mask(sample_image, sample_mask):
    """Test mask overlay"""
    # Test with default parameters
    overlay = overlay_mask(sample_image, sample_mask)
    assert overlay.shape == sample_image.shape
    assert overlay.dtype == np.uint8
    
    # Test with custom color and alpha
    overlay = overlay_mask(sample_image, sample_mask, color=(255, 0, 0), alpha=0.5)
    assert overlay.shape == sample_image.shape
    assert overlay.dtype == np.uint8
    
    # Verify that masked areas are colored
    masked_area = overlay[64:192, 64:192]
    assert not np.array_equal(masked_area, sample_image[64:192, 64:192])

def test_calculate_iou(sample_mask):
    """Test IoU calculation"""
    # Create a perfect match
    mask1 = sample_mask.copy()
    mask2 = sample_mask.copy()
    assert calculate_iou(mask1, mask2) == 1.0
    
    # Create a partial match
    mask2[96:192, 64:192] = 0
    iou = calculate_iou(mask1, mask2)
    assert 0 < iou < 1
    
    # Create no match
    mask2.fill(0)
    assert calculate_iou(mask1, mask2) == 0.0

def test_preprocess_image(sample_image):
    """Test image preprocessing"""
    # Test preprocessing
    processed = preprocess_image(sample_image)
    
    # Check output properties
    assert isinstance(processed, np.ndarray)
    assert processed.shape == sample_image.shape
    assert processed.dtype == np.float32
    
    # Check value range
    assert np.all(processed >= 0) and np.all(processed <= 1)

def test_postprocess_mask():
    """Test mask postprocessing"""
    # Create a sample probability mask
    prob_mask = np.random.rand(256, 256)
    
    # Test postprocessing
    binary_mask = postprocess_mask(prob_mask)
    
    # Check output properties
    assert isinstance(binary_mask, np.ndarray)
    assert binary_mask.shape == prob_mask.shape
    assert binary_mask.dtype == np.uint8
    
    # Check that mask is binary
    assert np.all(np.logical_or(binary_mask == 0, binary_mask == 255))

def test_image_utils_with_different_sizes():
    """Test image utilities with different image sizes"""
    # Create images of different sizes
    small_img = np.zeros((128, 128, 3), dtype=np.uint8)
    large_img = np.zeros((512, 512, 3), dtype=np.uint8)
    small_mask = np.zeros((128, 128), dtype=np.uint8)
    large_mask = np.zeros((512, 512), dtype=np.uint8)
    
    # Test overlay with different sizes
    overlay = overlay_mask(large_img, small_mask)
    assert overlay.shape == large_img.shape
    
    # Test IoU with different sizes
    iou = calculate_iou(small_mask, large_mask)
    assert 0 <= iou <= 1

def test_image_utils_with_invalid_input():
    """Test image utilities with invalid input"""
    # Test with None input
    assert load_image(None) is None
    
    # Test with invalid image type
    with pytest.raises(ValueError):
        overlay_mask(np.zeros((256, 256), dtype=np.float32), np.zeros((256, 256), dtype=np.uint8))
    
    # Test with invalid mask values
    with pytest.raises(ValueError):
        overlay_mask(
            np.zeros((256, 256, 3), dtype=np.uint8),
            np.ones((256, 256), dtype=np.uint8) * 128
        )
