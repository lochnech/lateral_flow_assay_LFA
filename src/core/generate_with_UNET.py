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

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
LOAD_CHECKPOINT_PATH = os.path.join(project_root, "data", "models", "model_checkpoint.pth.tar")
DATA_PATH = "./data"
INPUT_PATH = DATA_PATH + "/raw/test_images/"
OUTPUT_PATH = DATA_PATH + "/processed/masks/"

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

def pad_image(image, max_size=512):
    """Pad an image to make it square while maintaining aspect ratio.
    
    Args:
        image: Input image in RGB format
        max_size: Target size for both width and height
        
    Returns:
        Padded image in RGB format
    """
    # First resize while maintaining aspect ratio
    height, width = image.shape[:2]
    scale = min(max_size / width, max_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = cv2.resize(image, (new_width, new_height))
    
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
        value=tuple(map(int, get_background_color(image)))
    )
    
    return padded

# Function to generate mask
def generate_mask(image_path, save_path, model):
    # Load image first to get background color
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return
    
    # Convert to RGB for color analysis
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Pad the image
    padded = pad_image(image_rgb)
    
    # Create padded_images directory if it doesn't exist
    padded_dir = "./data/processed/padded_images/"
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
    output_folder = OUTPUT_PATH
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

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

    
    for filename in os.listdir(input_folder):
        if '.' in filename:
            image_path = os.path.join(input_folder, filename)
            save_path = os.path.join(output_folder, filename)
            print(f"\nProcessing image: {filename}")
            generate_mask(image_path, save_path, model)
        else:
            logging.warning(f"Skipping non-image file: {filename}")

if __name__ == "__main__":
    main()
    