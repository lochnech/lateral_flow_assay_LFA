import os
import sys
import pytest
import numpy as np
from pathlib import Path
import cv2

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from lfa.core.mask_application import apply_mask_to_image

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

def test_apply_mask_to_image(sample_image, sample_mask):
    """Test applying a mask to an image"""
    # Apply mask
    result = apply_mask_to_image(sample_image, sample_mask)
    
    # Check result properties
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8
    
    # Check that masked areas are preserved
    masked_area = result[64:192, 64:192]
    assert np.all(masked_area == sample_image[64:192, 64:192])
    
    # Check that non-masked areas are black
    non_masked_area = np.concatenate([
        result[:64, :],
        result[192:, :],
        result[64:192, :64],
        result[64:192, 192:]
    ])
    assert np.all(non_masked_area == 0)

def test_apply_mask_to_image_with_different_sizes():
    """Test applying a mask to an image with different sizes"""
    # Create image and mask with different sizes
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[32:96, 32:96] = 255
    
    # Apply mask
    result = apply_mask_to_image(image, mask)
    
    # Check that mask was resized correctly
    assert result.shape == image.shape

def test_apply_mask_to_image_with_invalid_input():
    """Test applying a mask with invalid input"""
    # Test with None input
    with pytest.raises(ValueError):
        apply_mask_to_image(None, np.zeros((256, 256), dtype=np.uint8))
    
    # Test with invalid mask type
    with pytest.raises(ValueError):
        apply_mask_to_image(
            np.zeros((256, 256, 3), dtype=np.uint8),
            np.zeros((256, 256), dtype=np.float32)
        )
    
    # Test with invalid mask values
    with pytest.raises(ValueError):
        apply_mask_to_image(
            np.zeros((256, 256, 3), dtype=np.uint8),
            np.ones((256, 256), dtype=np.uint8) * 128
        )

def test_mask_application_pipeline(tmp_path, sample_image, sample_mask):
    """Test the complete mask application pipeline"""
    # Create temporary directories
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    output_dir = tmp_path / "output"
    image_dir.mkdir()
    mask_dir.mkdir()
    output_dir.mkdir()
    
    # Save sample image and mask
    image_path = image_dir / "test.png"
    mask_path = mask_dir / "test.png"
    cv2.imwrite(str(image_path), sample_image)
    cv2.imwrite(str(mask_path), sample_mask)
    
    # Process all images
    for img_path in image_dir.glob("*.png"):
        mask_path = mask_dir / img_path.name
        if mask_path.exists():
            output_path = output_dir / img_path.name
            apply_mask_to_image(
                cv2.imread(str(img_path)),
                cv2.imread(str(mask_path), 0),
                str(output_path)
            )
    
    # Verify output
    assert len(list(output_dir.glob("*.png"))) == 1
    output_image = cv2.imread(str(output_dir / "test.png"))
    assert output_image is not None
    assert output_image.shape == sample_image.shape
