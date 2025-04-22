import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
from segmentation_ROI import UNET
from utils import load_checkpoint
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu" and config['device'] == "cuda":
    print("Warning: CUDA not available, using CPU instead")

# Constants from config
IMAGE_HEIGHT = config['image_height']
IMAGE_WIDTH = config['image_width']
LOAD_CHECKPOINT_PATH = config['model_checkpoint_path']
DATA_PATH = config['data_path']
INPUT_PATH = config['input_path']
OUTPUT_PATH = config['output_path']
PADDED_IMAGES_PATH = config['padded_images_path']
LOGS_PATH = config['logs_path']
TRANSFORMED_IMAGES_PATH = config['transformed_images_path']

def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs(LOGS_PATH, exist_ok=True)
    
    # Setup logging configuration
    logging.basicConfig(
        filename=f'{LOGS_PATH}/transform_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def log_transformed_image(image, transformed_image, image_name):
    """Save original and transformed images for inspection"""
    os.makedirs(TRANSFORMED_IMAGES_PATH, exist_ok=True)
    
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
    plt.savefig(f'{TRANSFORMED_IMAGES_PATH}/{image_name}_comparison.png')
    plt.close()
    
    # Log the transformation
    logging.info(f"Processed image {image_name}")
    logging.info(f"Original shape: {image.shape}")
    logging.info(f"Transformed shape: {transformed_np.shape}")

def generate_transformed_data(image_path):
    """Generate and log transformed data with padding"""
    setup_logging()
    
    # Define transform with padding from config
    transform = A.Compose([
        A.LongestMaxSize(max_size=config['transform']['max_size']),
        A.PadIfNeeded(
            min_height=config['transform']['padding']['min_height'],
            min_width=config['transform']['padding']['min_width'],
            border_mode=getattr(cv2, f"BORDER_{config['transform']['padding']['border_mode'].upper()}"),
            value=config['transform']['padding']['value']
        ),
        A.Normalize(
            mean=config['transform']['normalize']['mean'],
            std=config['transform']['normalize']['std'],
            max_pixel_value=config['transform']['normalize']['max_pixel_value'],
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
model = UNET(
    in_channels=config['model']['in_channels'],
    out_channels=config['model']['out_channels']
).to(DEVICE)
load_checkpoint(torch.load(LOAD_CHECKPOINT_PATH, map_location=DEVICE), model)
model.eval()

def generate_mask(image_path, save_path):
    # Define the transformation pipeline from config
    transform = A.Compose([
        A.LongestMaxSize(max_size=config['transform']['max_size']),
        A.PadIfNeeded(
            min_height=config['transform']['padding']['min_height'],
            min_width=config['transform']['padding']['min_width'],
            border_mode=getattr(cv2, f"BORDER_{config['transform']['padding']['border_mode'].upper()}"),
            value=config['transform']['padding']['value']
        ),
        A.Normalize(
            mean=config['transform']['normalize']['mean'],
            std=config['transform']['normalize']['std'],
            max_pixel_value=config['transform']['normalize']['max_pixel_value'],
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
    transformed_image = ((transformed_image * config['transform']['normalize']['std'] + 
                         config['transform']['normalize']['mean']) * 
                        config['transform']['normalize']['max_pixel_value']).astype(np.uint8)
    # Convert back to BGR for saving
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
    
    # Create padded_images directory if it doesn't exist
    os.makedirs(PADDED_IMAGES_PATH, exist_ok=True)
    
    # Save transformed image
    padded_path = os.path.join(PADDED_IMAGES_PATH, os.path.basename(image_path))
    cv2.imwrite(padded_path, transformed_image)
    print(f"Saved transformed image to: {padded_path}")
    
    # Continue with model processing
    input_tensor = augmented["image"].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.sigmoid(output).cpu().numpy().squeeze()
        mask = (mask > config['mask']['threshold']).astype(np.uint8) * config['mask']['binary_value']
    
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
    