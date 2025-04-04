import cv2
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path

def load_image(image_path: str) -> np.ndarray:
    """Load an image from file
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        np.ndarray: Loaded image
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save_image(image: np.ndarray, output_path: str) -> None:
    """Save an image to file
    
    Args:
        image (np.ndarray): Image to save
        output_path (str): Path to save the image
    """
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to BGR for saving
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, image)

def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize an image to the specified size
    
    Args:
        image (np.ndarray): Input image
        size (tuple): Target size (width, height)
        
    Returns:
        np.ndarray: Resized image
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        np.ndarray: Normalized image
    """
    return image.astype(np.float32) / 255.0

def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Denormalize image from [0, 1] range to [0, 255]
    
    Args:
        image (np.ndarray): Normalized image
        
    Returns:
        np.ndarray: Denormalized image
    """
    return (image * 255).astype(np.uint8)

def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0), 
                alpha: float = 0.5) -> np.ndarray:
    """Overlay a mask on an image
    
    Args:
        image (np.ndarray): Input image
        mask (np.ndarray): Binary mask
        color (tuple): RGB color for the mask
        alpha (float): Opacity of the overlay
        
    Returns:
        np.ndarray: Image with mask overlay
    """
    # Ensure mask is binary
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    
    # Create color overlay
    overlay = np.zeros_like(image)
    overlay[mask > 0] = color
    
    # Blend images
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return result

def visualize_prediction(image: np.ndarray, mask: np.ndarray, 
                        save_path: Optional[str] = None) -> None:
    """Visualize image with predicted mask
    
    Args:
        image (np.ndarray): Input image
        mask (np.ndarray): Predicted mask
        save_path (str, optional): Path to save the visualization
    """
    # Create figure
    plt.figure(figsize=(12, 4))
    
    # Plot original image
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot mask
    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
    
    # Plot overlay
    plt.subplot(133)
    overlay = overlay_mask(image, mask)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def preprocess_image(image: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Preprocess image for model input
    
    Args:
        image (np.ndarray): Input image
        size (tuple): Target size for resizing
        
    Returns:
        np.ndarray: Preprocessed image
    """
    # Resize
    image = resize_image(image, size)
    
    # Normalize
    image = normalize_image(image)
    
    # Convert to float32
    image = image.astype(np.float32)
    
    return image

def postprocess_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Postprocess model output mask
    
    Args:
        mask (np.ndarray): Model output mask
        threshold (float): Threshold for binarization
        
    Returns:
        np.ndarray: Postprocessed mask
    """
    # Apply threshold
    mask = (mask > threshold).astype(np.uint8) * 255
    
    # Remove small objects
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate Intersection over Union between two masks
    
    Args:
        mask1 (np.ndarray): First mask
        mask2 (np.ndarray): Second mask
        
    Returns:
        float: IoU score
    """
    # Ensure masks are binary
    mask1 = (mask1 > 127).astype(np.uint8)
    mask2 = (mask2 > 127).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    return intersection / union
