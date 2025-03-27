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
OUTPUT_PATH = DATA_PATH + "/result_images/"

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
    background_color = get_background_color(image_rgb)
    
    # First resize while maintaining aspect ratio
    max_size = 512
    height, width = image_rgb.shape[:2]
    scale = min(max_size / width, max_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = cv2.resize(image_rgb, (new_width, new_height))
    
    # Calculate padding
    pad_height = max_size - new_height
    pad_width = max_size - new_width
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
    
    # Create padded_images directory if it doesn't exist
    padded_dir = "./data/padded_images/"
    os.makedirs(padded_dir, exist_ok=True)
    
    # Save padded image
    padded_path = os.path.join(padded_dir, os.path.basename(image_path))
    cv2.imwrite(padded_path, cv2.cvtColor(padded, cv2.COLOR_RGB2BGR))
    print(f"Saved padded image to: {padded_path}")
    
    # Convert to tensor for model
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
        mask = (mask > 0.5).astype(np.uint8) * 255  # Thresholding for binary mask
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, mask)
    print(f"Saved mask to: {save_path}")

def main():
    input_folder = INPUT_PATH
    output_folder = OUTPUT_PATH + "result_masks/"
    
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
    