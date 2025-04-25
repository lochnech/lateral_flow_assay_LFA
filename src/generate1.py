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
RAW_IMAGES_PATH = config['raw_images_path']
MASKS_PATH = config['masks_path']
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

def get_preprocessing_transform():
    """Create and return the preprocessing transformation pipeline from config"""
    return A.Compose([
        A.LongestMaxSize(max_size=config['preprocessing']['transform']['max_size']),
        A.PadIfNeeded(
            min_height=config['preprocessing']['transform']['padding']['min_height'],
            min_width=config['preprocessing']['transform']['padding']['min_width'],
            border_mode=getattr(cv2, f"BORDER_{config['preprocessing']['transform']['padding']['border_mode'].upper()}"),
            value=config['preprocessing']['transform']['padding']['value']
        ),
        A.Normalize(
            mean=config['preprocessing']['transform']['normalize']['mean'],
            std=config['preprocessing']['transform']['normalize']['std'],
            max_pixel_value=config['preprocessing']['transform']['normalize']['max_pixel_value'],
        ),
        ToTensorV2(),
    ])

def preprocess_image(image_path):
    """Load and preprocess an image according to the config settings"""
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None, None, None
        
    # Read image with error checking
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return None, None, None
        
    # Convert to RGB for transformations
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get transformation pipeline and apply it
    transform = get_preprocessing_transform()
    augmented = transform(image=image_rgb)
    transformed_tensor = augmented["image"]
    
    # Create output version of transformed image for saving
    transformed_image = transformed_tensor.permute(1, 2, 0).cpu().numpy()
    # Denormalize
    transformed_image = ((transformed_image * config['preprocessing']['transform']['normalize']['std'] + 
                         config['preprocessing']['transform']['normalize']['mean']) * 
                        config['preprocessing']['transform']['normalize']['max_pixel_value']).astype(np.uint8)
    # Convert back to BGR for saving
    transformed_image_bgr = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
    
    # Log image info
    image_name = os.path.basename(image_path).split('.')[0]
    logging.info(f"Processed image {image_name}")
    logging.info(f"Original shape: {image_rgb.shape}")
    logging.info(f"Transformed shape: {transformed_tensor.shape}")
    
    return transformed_tensor, transformed_image_bgr, image_rgb

def generate_transformed_data(image_path):
    """Generate and log transformed data with padding"""
    setup_logging()
    
    # Preprocess the image
    transformed_tensor, transformed_image_bgr, original_image = preprocess_image(image_path)
    if transformed_tensor is None:
        return None
    
    image_name = os.path.basename(image_path).split('.')[0]
    
    try:
        # Log the transformation
        log_transformed_image(original_image, transformed_tensor, image_name)
        
        # Log tensor statistics
        logging.info(f"Tensor stats for {image_name}:")
        logging.info(f"Min value: {transformed_tensor.min()}")
        logging.info(f"Max value: {transformed_tensor.max()}")
        logging.info(f"Mean value: {transformed_tensor.mean()}")
        logging.info(f"Std value: {transformed_tensor.std()}")
        
        return transformed_tensor
        
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

def process_mask(mask):
    """Apply post-processing to the mask based on config settings"""
    # Apply Gaussian blur if enabled
    if config['mask']['processing']['gaussian_blur']['apply']:
        kernel_size = config['mask']['processing']['gaussian_blur']['kernel_size']
        sigma = config['mask']['processing']['gaussian_blur']['sigma']
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
        # Re-threshold after blur to get binary mask again
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Apply dilation if enabled
    if config['mask']['processing']['dilation']['apply']:
        kernel_size = config['mask']['processing']['dilation']['kernel_size']
        iterations = config['mask']['processing']['dilation']['iterations']
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=iterations)
    
    # Apply morphological operations if enabled
    if config['mask']['processing']['morphology']['apply']:
        operation = config['mask']['processing']['morphology']['operation']
        kernel_size = config['mask']['processing']['morphology']['kernel_size']
        iterations = config['mask']['processing']['morphology']['iterations']
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if operation == "open":
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == "close":
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    return mask

def generate_mask(image_path, save_path):
    # Preprocess the image
    transformed_tensor, transformed_image_bgr, original_image = preprocess_image(image_path)
    if transformed_tensor is None:
        return
    
    # Create padded_images directory if it doesn't exist
    os.makedirs(PADDED_IMAGES_PATH, exist_ok=True)
    
    # Save transformed image
    padded_path = os.path.join(PADDED_IMAGES_PATH, os.path.basename(image_path))
    cv2.imwrite(padded_path, transformed_image_bgr)
    print(f"Saved transformed image to: {padded_path}")
    
    # Continue with model processing
    input_tensor = transformed_tensor.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.sigmoid(output).cpu().numpy().squeeze()
        mask = (mask > config['mask']['threshold']).astype(np.uint8) * config['mask']['binary_value']
        
        # Apply post-processing to the mask
        mask = process_mask(mask)
    
    os.makedirs(MASKS_PATH, exist_ok=True)
    cv2.imwrite(save_path, mask)
    print(f"Saved mask to: {save_path}")
    
    # Also save a debug image showing the mask overlay
    debug_dir = os.path.join(os.path.dirname(save_path), "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Create a colored version of the mask for visualization
    debug_image = transformed_image_bgr.copy()
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask_rgb[mask > 0] = [0, 0, 255]  # Red color in BGR where mask is white
    
    # Add semi-transparent overlay
    debug_image = cv2.addWeighted(mask_rgb, 0.3, debug_image, 0.7, 0)
    
    # Save debug image
    debug_path = os.path.join(debug_dir, f"debug_{os.path.basename(image_path)}")
    cv2.imwrite(debug_path, debug_image)
    print(f"Saved debug mask overlay to: {debug_path}")

def is_valid_image_file(filename):
    """Check if a file is a valid image file based on its extension"""
    # Get valid file extensions from config
    valid_extensions = config['preprocessing']['file_extensions']
    
    # Check if filename ends with any valid extension (case insensitive)
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

def main():
    # Setup logging
    setup_logging()
    
    for filename in os.listdir(RAW_IMAGES_PATH):
        if is_valid_image_file(filename):
            image_path = os.path.join(RAW_IMAGES_PATH, filename)
            save_path = os.path.join(MASKS_PATH, filename)
            generate_mask(image_path, save_path)
        else:
            logging.warning(f"Skipping non-image file: {filename}")

if __name__ == "__main__":
    main()
    