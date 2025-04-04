import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
import sys
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from lfa.models.unet import UNET
from lfa.utils.utils import load_checkpoint

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
LOAD_CHECKPOINT_PATH = os.path.join(project_root, "models", "model_checkpoint.pth.tar")
DATA_PATH = "./data"
INPUT_PATH = DATA_PATH + "/test_images/"
OUTPUT_PATH = DATA_PATH + "/masks/"

def setup_logging():
    """Set up logging configuration"""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename=f'logs/transform_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def log_transformed_image(image, transformed_image, image_name):
    """Save original and transformed images for inspection"""
    os.makedirs('logs/transformed_images', exist_ok=True)
    
    if torch.is_tensor(transformed_image):
        transformed_np = transformed_image.numpy().transpose(1, 2, 0)
    else:
        transformed_np = transformed_image
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(transformed_np)
    ax2.set_title('Transformed')
    ax2.axis('off')
    plt.savefig(f'logs/transformed_images/{image_name}_comparison.png')
    plt.close()
    
    logging.info(f"Processed image {image_name}")
    logging.info(f"Original shape: {image.shape}")
    logging.info(f"Transformed shape: {transformed_np.shape}")

def get_background_color(image):
    """Extract the background color from the image edges"""
    edges = np.concatenate([
        image[0, :],
        image[-1, :],
        image[:, 0],
        image[:, -1]
    ])
    edges_reshaped = edges.reshape(-1, 3)
    unique_colors, counts = np.unique(edges_reshaped, axis=0, return_counts=True)
    return unique_colors[np.argmax(counts)]

def create_smooth_transition(image, padded_image, background_color):
    """Create a smooth transition between the original image and padding"""
    height, width = image.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    kernel_size = 25
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    blur_size = 35
    blurred_mask = cv2.GaussianBlur(eroded, (blur_size, blur_size), 0)
    blurred_mask = cv2.GaussianBlur(blurred_mask, (blur_size, blur_size), 0)
    blurred_mask = blurred_mask.astype(float) / 255
    blurred_mask = np.power(blurred_mask, 0.7)
    blurred_mask = np.stack([blurred_mask] * 3, axis=-1)
    
    padded_height, padded_width = padded_image.shape[:2]
    top_pad = (padded_height - height) // 2
    left_pad = (padded_width - width) // 2
    
    result = padded_image.copy()
    result[top_pad:top_pad+height, left_pad:left_pad+width] = (
        image * blurred_mask + 
        background_color * (1 - blurred_mask)
    ).astype(np.uint8)
    
    return result

def enhance_contrast(image):
    """Enhance image contrast using CLAHE"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16,16))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

def generate_transformed_data(image_path):
    """Generate and log transformed data with padding"""
    setup_logging()
    
    image = np.array(Image.open(image_path).convert("RGB"))
    background_color = get_background_color(image)
    
    transform = A.Compose([
        A.LongestMaxSize(max_size=512),
        A.PadIfNeeded(
            min_height=512,
            min_width=512,
            border_mode=cv2.BORDER_CONSTANT,
            value=tuple(map(int, background_color))
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
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        log_transformed_image(image, transformed_image, image_name)
        
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

def generate_mask(image_path, save_path):
    """Generate mask for an image using the UNET model"""
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
    
    # Generate transformed data
    transformed_image = generate_transformed_data(image_path)
    
    # Add batch dimension
    transformed_image = transformed_image.unsqueeze(0).to(DEVICE)
    
    # Generate mask
    with torch.no_grad():
        mask = model(transformed_image)
        mask = torch.sigmoid(mask)
        mask = (mask > 0.5).float()
    
    # Convert mask to numpy and remove batch dimension
    mask = mask.squeeze().cpu().numpy()
    
    # Save mask
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, (mask * 255).astype(np.uint8))
    print(f"Saved mask to: {save_path}")

def main():
    """Main function to process all images in the input directory"""
    # Load model
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    
    # Load checkpoint
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
    
    # Process all images in the input directory
    for filename in os.listdir(INPUT_PATH):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(INPUT_PATH, filename)
            mask_path = os.path.join(OUTPUT_PATH, filename)
            generate_mask(image_path, mask_path)

if __name__ == "__main__":
    main()
