import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import cv2
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.UNETpp.segmentation_ROIpp import UNETpp



def setup_logging():
    """Sets up logging configuration for transformation results"""
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
        
        # Log tensor statistics
        logging.info(f"Tensor stats for {image_name}:")
        logging.info(f"Min value: {transformed_image.min()}")
        logging.info(f"Max value: {transformed_image.max()}")
        logging.info(f"Mean value: {transformed_image.mean()}")
        logging.info(f"Std value: {transformed_image.std()}")
        
        return transformed_image
        
    except Exception as e:
        logging.error(f"Error processing {image_name}: {str(e)}")
        raise e

def generate_mask(image_path, model_path, save_path):
    """Generate segmentation mask using UNET++ model"""
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = UNETpp(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    # Get transformed tensor
    transformed_tensor = generate_transformed_data(image_path)
    
    # Add batch dimension and send to device
    input_tensor = transformed_tensor.unsqueeze(0).to(device)
    
    # Get predictions (UNET++ returns multiple outputs)
    with torch.no_grad():
        outputs = model(input_tensor)
        # Use the final output (last element in the list)
        mask = torch.sigmoid(outputs[-1]).cpu().numpy().squeeze()
        mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save mask
    cv2.imwrite(save_path, mask)
    print(f"Mask saved to {save_path}")
    
    return mask

def main():
    """Main function to process images"""
    # Example usage
    INPUT_PATH = "data/test_images/"
    OUTPUT_PATH = "data/result_masks/UNETpp/"
    model_path = "models/unetpp_checkpoint.pth.tar"
    save_path = "outputs/masks/test_mask.png"
    for filename in os.listdir(INPUT_PATH):
        try:
            if '.' in filename:
                image_path = os.path.join(INPUT_PATH, filename)
                save_path = os.path.join(OUTPUT_PATH, filename)
                generate_mask(image_path, model_path, save_path)
            else:
                logging.warning(f"Skipping non-image file: {filename}")
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
            raise e

if __name__ == "__main__":
    main()
