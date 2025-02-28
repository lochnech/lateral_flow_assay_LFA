import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
from src.UNET.segmentation_ROI import UNET
from src.utils import load_checkpoint
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
LOAD_CHECKPOINT_PATH = "./models/model_checkpoint.pth.tar"
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

def generate_transformed_data(image_path):
    """Generate and log transformed data with padding"""
    setup_logging()
    
    # Define transform with padding
    transform = A.Compose([
        # First resize to maintain aspect ratio
        A.LongestMaxSize(max_size=512),
        # Then pad to get 512x512
        A.PadIfNeeded(
            min_height=512,
            min_width=512,
            border_mode=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # Black padding
        ),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    
    # Load and transform image
    image = np.array(Image.open(image_path).convert("RGB"))
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
load_checkpoint(torch.load(LOAD_CHECKPOINT_PATH, map_location=DEVICE), model)
model.eval()

# Function to generate mask
def generate_mask(image_path, save_path):
    # Define the transformation pipeline
    transform = A.Compose([
        # First resize to maintain aspect ratio
        A.LongestMaxSize(max_size=512),
        # Then pad to get 512x512
        A.PadIfNeeded(
            min_height=512,
            min_width=512,
            border_mode=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # Black padding
        ),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
        
    # Read image with error checking
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return
        
    # Convert to RGB for transformations
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transformations
    augmented = transform(image=image)
    
    # Save transformed image before model processing
    transformed_image = augmented["image"].permute(1, 2, 0).cpu().numpy()
    # Denormalize
    transformed_image = ((transformed_image * [1.0, 1.0, 1.0] + [0.0, 0.0, 0.0]) * 255).astype(np.uint8)
    # Convert back to BGR for saving
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
    
    # Create padded_images directory if it doesn't exist
    padded_dir = "./data/padded_images/"
    os.makedirs(padded_dir, exist_ok=True)
    
    # Save transformed image
    padded_path = os.path.join(padded_dir, os.path.basename(image_path))
    cv2.imwrite(padded_path, transformed_image)
    print(f"Saved transformed image to: {padded_path}")
    
    # Continue with model processing
    input_tensor = augmented["image"].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.sigmoid(output).cpu().numpy().squeeze()
        mask = (mask > 0.5).astype(np.uint8) * 255  # Thresholding for binary mask
    
    cv2.imwrite(save_path, mask)
    print(f"Saved mask to: {save_path}")

def main():
    input_folder = INPUT_PATH
    output_folder = OUTPUT_PATH + "result_masks/"
    for filename in os.listdir(input_folder):
        if '.' in filename:
            image_path = os.path.join(input_folder, filename)
            save_path = os.path.join(output_folder, filename)
            generate_mask(image_path, save_path)
        else:
            logging.warning(f"Skipping non-image file: {filename}")

if __name__ == "__main__":
    main()
    