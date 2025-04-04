import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.UNET.segmentation_ROI import UNET
from src.utils.utils import load_checkpoint
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
LOAD_CHECKPOINT_PATH = os.path.join(project_root, "src", "UNET", "models", "model_checkpoint.pth.tar")
DATA_PATH = "./data"
INPUT_PATH = DATA_PATH + "/test_images/"
OUTPUT_PATH = DATA_PATH + "/masks/"

def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Setup logging configuration
    logging.basicConfig(
        filename=f'logs/transform_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def log_transformed_image(image, transformed_image, image_name):
    """Save original and transformed images for inspection"""
    os.makedirs('logs/transformed_images', exist_ok=True)
    
    # Convert tensor to numpy for visualization
    if torch.is_tensor(transformed_image):
        transformed_np = transformed_image.numpy().transpose(1, 2, 0)
    else:
        transformed_np = transformed_image
    
    # Create subplot with original and transformed
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot original
    ax1.imshow(image)
    ax1.set_title('Original')
    ax1.axis('off')
    
    # Plot transformed
    ax2.imshow(transformed_np)
    ax2.set_title('Transformed')
    ax2.axis('off')
    
    # Save plot
    plt.savefig(f'logs/transformed_images/{image_name}_comparison.png')
    plt.close()
    
    # Log the transformation
    logging.info(f"Processed image {image_name}")
    logging.info(f"Original shape: {image.shape}")
    logging.info(f"Transformed shape: {transformed_np.shape}")

def get_background_color(image):
    """Extract the background color from the image edges"""
    # Sample from the edges of the image
    edges = np.concatenate([
        image[0, :],  # top edge
        image[-1, :],  # bottom edge
        image[:, 0],  # left edge
        image[:, -1]  # right edge
    ])
    
    # Get the most common color (mode) from the edges
    # Reshape to 2D array of pixels
    edges_reshaped = edges.reshape(-1, 3)
    # Get unique colors and their counts
    unique_colors, counts = np.unique(edges_reshaped, axis=0, return_counts=True)
    # Get the most common color
    background_color = unique_colors[np.argmax(counts)]
    
    return background_color

def create_smooth_transition(image, padded_image, background_color):
    """Create a smooth transition between the original image and padding using morphological operations"""
    # Create a mask for the original image area
    height, width = image.shape[:2]
    
    # Create initial mask
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Create kernel for morphological operations
    kernel_size = 25
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply morphological operations to create a more gradual transition
    # First dilate to expand the mask
    dilated = cv2.dilate(mask, kernel, iterations=2)
    
    # Then erode to create a smoother edge
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # Apply Gaussian blur with larger kernel for smoother transition
    blur_size = 35  # Increased blur size
    blurred_mask = cv2.GaussianBlur(eroded, (blur_size, blur_size), 0)
    
    # Apply additional smoothing to make the transition even more gradual
    blurred_mask = cv2.GaussianBlur(blurred_mask, (blur_size, blur_size), 0)
    
    # Normalize the blurred mask to 0-1 range
    blurred_mask = blurred_mask.astype(float) / 255
    
    # Apply a power function to make the transition more gradual
    blurred_mask = np.power(blurred_mask, 0.7)  # Adjusted power for smoother transition
    
    # Expand mask to 3 channels
    blurred_mask = np.stack([blurred_mask] * 3, axis=-1)
    
    # Calculate padding offsets
    padded_height, padded_width = padded_image.shape[:2]
    top_pad = (padded_height - height) // 2
    left_pad = (padded_width - width) // 2
    
    # Create the blended result
    result = padded_image.copy()
    
    # Place the blended image at the correct position with padding offsets
    result[top_pad:top_pad+height, left_pad:left_pad+width] = (
        image * blurred_mask + 
        background_color * (1 - blurred_mask)
    ).astype(np.uint8)
    
    return result

def enhance_contrast(image):
    """Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split into channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel with more conservative parameters
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16,16))  # Reduced clip limit, increased tile size
    l = clahe.apply(l)
    
    # Merge channels back
    enhanced_lab = cv2.merge((l, a, b))
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return enhanced

def generate_transformed_data(image_path):
    """Generate and log transformed data with padding"""
    setup_logging()
    
    # Load image first to get background color
    image = np.array(Image.open(image_path).convert("RGB"))
    background_color = get_background_color(image)
    
    # Define transform with padding
    transform = A.Compose([
        # First resize to maintain aspect ratio
        A.LongestMaxSize(max_size=512),
        # Then pad to get 512x512 using background color
        A.PadIfNeeded(
            min_height=512,
            min_width=512,
            border_mode=cv2.BORDER_CONSTANT,
            value=tuple(map(int, background_color))  # Convert to tuple of integers
        ),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    
    image_name = os.path.basename(image_path).split('.')[0]
    
    try:
        # Apply transformation
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        
        # Log the transformation
        log_transformed_image(image, transformed_image, image_name)
        
        # Log tensor statistics and padding info
        logging.info(f"Tensor stats for {image_name}:")
        logging.info(f"Original shape: {image.shape}")
        logging.info(f"Transformed shape: {transformed_image.shape}")
        logging.info(f"Background color used: {background_color}")
        logging.info(f"Min value: {transformed_image.min()}")
        logging.info(f"Max value: {transformed_image.max()}")
        logging.info(f"Mean value: {transformed_image.mean()}")
        logging.info(f"Std value: {transformed_image.std()}")
        
        return transformed_image
        
    except Exception as e:
        logging.error(f"Error processing {image_name}: {str(e)}")
        raise e

# Load model
model = UNET(in_channels=3, out_channels=1).to(DEVICE)

# Load checkpoint with error handling
try:
    if not os.path.exists(LOAD_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint file not found at: {LOAD_CHECKPOINT_PATH}")
    
    print(f"Loading checkpoint from: {LOAD_CHECKPOINT_PATH}")
    checkpoint = torch.load(LOAD_CHECKPOINT_PATH, map_location=DEVICE)
    load_checkpoint(checkpoint, model)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading checkpoint: {str(e)}")
    print("Please ensure you have trained the model and the checkpoint file exists.")
    sys.exit(1)

# Function to generate mask
def generate_mask(image_path, save_path):
    # Load image first to get background color
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return
    
    # Convert to RGB for color analysis
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Enhance contrast
    enhanced = enhance_contrast(image_rgb)
    
    # Get background color from enhanced image
    background_color = get_background_color(enhanced)
    
    # Print background color for debugging
    print(f"Background color detected: {background_color}")
    
    # First resize while maintaining aspect ratio
    # Use a larger target size to preserve more details
    target_size = 768  # Increased from 512 to 768
    height, width = enhanced.shape[:2]
    
    # Calculate scale factors for both dimensions
    scale = min(target_size / width, target_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Use high-quality interpolation for resizing
    resized = cv2.resize(enhanced, (new_width, new_height), 
                        interpolation=cv2.INTER_LANCZOS4)
    
    # Calculate padding
    pad_height = target_size - new_height
    pad_width = target_size - new_width
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    
    # Apply padding with background color
    padded = cv2.copyMakeBorder(
        resized,
        top_pad, bottom_pad, left_pad, right_pad,
        cv2.BORDER_CONSTANT,
        value=tuple(map(int, background_color))
    )
    
    # Create smooth transition
    padded = create_smooth_transition(resized, padded, background_color)
    
    # Create padded_images directory if it doesn't exist
    padded_dir = "./data/padded_images/"
    os.makedirs(padded_dir, exist_ok=True)
    
    # Save padded image
    padded_path = os.path.join(padded_dir, os.path.basename(image_path))
    cv2.imwrite(padded_path, cv2.cvtColor(padded, cv2.COLOR_RGB2BGR))
    print(f"Saved padded image to: {padded_path}")
    
    # Continue with model processing
    transform = A.Compose([
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    
    # Apply normalization
    transformed = transform(image=padded)
    input_tensor = transformed["image"].unsqueeze(0).to(DEVICE)
    
    # Generate mask
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.sigmoid(output).cpu().numpy().squeeze()
        mask = (mask > 0.35).astype(np.uint8) * 255  # Increased threshold from 0.1 to 0.35
    
    # Resize mask back to original dimensions
    mask = cv2.resize(mask, (width, height), 
                     interpolation=cv2.INTER_LINEAR)
    
    # Apply more conservative morphological operations
    kernel = np.ones((2,2), np.uint8)  # Smaller kernel
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # Single iteration
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # Single iteration
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, mask)
    print(f"Saved mask to: {save_path}")

def main():
    input_folder = INPUT_PATH
    output_folder = OUTPUT_PATH
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if '.' in filename:
            image_path = os.path.join(input_folder, filename)
            save_path = os.path.join(output_folder, filename)
            print(f"\nProcessing image: {filename}")
            generate_mask(image_path, save_path)
        else:
            logging.warning(f"Skipping non-image file: {filename}")

if __name__ == "__main__":
    main()
    