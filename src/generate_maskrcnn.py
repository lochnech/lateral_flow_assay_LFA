import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2
import numpy as np
import os
import yaml
import logging
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu" and config['device'] == "cuda":
    print("Warning: CUDA not available, using CPU instead")

# Constants from config
LOGS_PATH = config['logs_path']
TRANSFORMED_IMAGES_PATH = config['transformed_images_path']
MASKS_PATH = config['maskrcnn']['masks_path']
APPLIED_MASKS_PATH = config['maskrcnn']['applied_masks_path']

def setup_logging():
    """Setup logging configuration"""
    os.makedirs(LOGS_PATH, exist_ok=True)
    logging.basicConfig(
        filename=f'{LOGS_PATH}/maskrcnn_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_model():
    """Load and configure the Mask R-CNN model"""
    # Load pre-trained model
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Modify the model to handle our specific case
    # Set a higher confidence threshold for detection
    model.roi_heads.score_thresh = 0.2  # Higher threshold for detection
    model.roi_heads.nms_thresh = 0.3    # Higher NMS threshold to reduce overlapping detections
    
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image_path):
    """Load and preprocess an image for Mask R-CNN"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return None, None
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 1. Noise reduction
    denoised = cv2.fastNlMeansDenoisingColored(image_rgb, None, 10, 10, 7, 21)
    
    # 2. Convert to LAB color space for better color processing
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # 3. Enhance contrast in L channel
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # 4. Enhance color channels
    a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
    b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    
    # 5. Merge enhanced channels
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # 6. Sharpen the image
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced_rgb, -1, kernel)
    
    # 7. Edge enhancement
    edges = cv2.Canny(sharpened, 100, 200)
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # 8. Combine original and edge information
    final_image = cv2.addWeighted(sharpened, 0.7, edges, 0.3, 0)
    
    # Save debug visualization of preprocessing steps
    debug_dir = os.path.join(os.path.dirname(image_path), "preprocessing_debug")
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(debug_dir, f"preprocess_{os.path.basename(image_path)}")
    
    # Create a visualization of all preprocessing steps
    steps = [
        ("Original", image_rgb),
        ("Denoised", denoised),
        ("Contrast Enhanced", enhanced_rgb),
        ("Sharpened", sharpened),
        ("Edges", edges),
        ("Final", final_image)
    ]
    
    # Create a grid of images
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    for idx, (title, img) in enumerate(steps):
        row = idx // cols
        col = idx % cols
        axes[row, col].imshow(img)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(debug_path)
    plt.close()
    
    # Convert to PIL Image
    image_pil = Image.fromarray(final_image)
    
    # Convert to tensor and normalize
    image_tensor = F.to_tensor(image_pil)
    
    return image_tensor, image_rgb

def process_predictions(predictions, image_shape):
    """Process model predictions into a binary mask"""
    # Initialize empty mask
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    if len(predictions[0]['masks']) > 0:
        # Get all masks with confidence > 0.2
        valid_masks = []
        valid_scores = []
        
        for i, score in enumerate(predictions[0]['scores']):
            if score > 0.2:  # Higher threshold for mask selection
                valid_masks.append(predictions[0]['masks'][i, 0].cpu().numpy())
                valid_scores.append(score.item())
        
        if valid_masks:
            # Combine all valid masks
            combined_mask = np.zeros_like(mask, dtype=np.float32)
            for mask_array in valid_masks:
                combined_mask = np.maximum(combined_mask, mask_array)
            
            # Convert to binary mask with higher threshold
            mask = (combined_mask > 0.4).astype(np.uint8) * 255  # Higher threshold for mask binarization
            
            # Additional post-processing for the combined mask
            # Fill holes in the mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            # Remove small artifacts
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    return mask

def apply_post_processing(mask):
    """Apply post-processing to the mask based on config settings"""
    # Apply Gaussian blur if enabled
    if config['mask']['processing']['gaussian_blur']['apply']:
        kernel_size = config['mask']['processing']['gaussian_blur']['kernel_size']
        sigma = config['mask']['processing']['gaussian_blur']['sigma']
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
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

def save_debug_visualization(image, mask, predictions, save_path):
    """Save debug visualization of the mask overlay and detections"""
    debug_image = image.copy()
    
    # Draw mask overlay
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask_rgb[mask > 0] = [0, 0, 255]  # Red color in BGR where mask is white
    debug_image = cv2.addWeighted(mask_rgb, 0.3, debug_image, 0.7, 0)
    
    # Draw bounding boxes and scores
    if len(predictions[0]['boxes']) > 0:
        for box, score in zip(predictions[0]['boxes'], predictions[0]['scores']):
            if score > 0.2:  # Only draw boxes with confidence > 0.2
                box = box.cpu().numpy().astype(int)
                cv2.rectangle(debug_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(debug_image, f'{score:.2f}', (box[0], box[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save debug image
    cv2.imwrite(save_path, debug_image)

def apply_mask_to_image(image, mask):
    """Apply the mask to the image using the background color from config"""
    # Create a copy of the image
    result = image.copy()
    
    # Get background color from config
    bg_color = np.array(config['mask_application']['background_color'], dtype=np.uint8)
    
    # Apply mask: keep original image where mask is white, use background color elsewhere
    result[mask == 0] = bg_color
    
    return result

def generate_mask(image_path, save_path):
    """Generate mask using Mask R-CNN model"""
    # Load and preprocess image
    image_tensor, image_rgb = preprocess_image(image_path)
    if image_tensor is None:
        return
    
    # Get model predictions
    with torch.no_grad():
        predictions = model([image_tensor.to(DEVICE)])
    
    # Process predictions into mask
    mask = process_predictions(predictions, image_rgb.shape)
    
    # Apply post-processing
    mask = apply_post_processing(mask)
    
    # Save mask
    os.makedirs(MASKS_PATH, exist_ok=True)
    cv2.imwrite(save_path, mask)
    
    # Apply mask to image and save
    os.makedirs(APPLIED_MASKS_PATH, exist_ok=True)
    applied_mask_path = os.path.join(APPLIED_MASKS_PATH, os.path.basename(image_path))
    result_image = apply_mask_to_image(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), mask)
    cv2.imwrite(applied_mask_path, result_image)
    
    # Save debug visualization
    debug_dir = os.path.join(os.path.dirname(save_path), "debug")
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(debug_dir, f"debug_{os.path.basename(image_path)}")
    save_debug_visualization(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), mask, predictions, debug_path)
    
    # Log results
    logging.info(f"Processed image: {os.path.basename(image_path)}")
    if len(predictions[0]['scores']) > 0:
        confidence = predictions[0]['scores'][0].item()
        logging.info(f"Detection confidence: {confidence:.3f}")

def is_valid_image_file(filename):
    """Check if a file is a valid image file based on its extension"""
    valid_extensions = config['preprocessing']['file_extensions']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

def main():
    # Setup logging
    setup_logging()
    
    # Load model
    global model
    model = load_model()
    
    # Process all images in the raw images directory
    for filename in os.listdir(config['raw_images_path']):
        if is_valid_image_file(filename):
            image_path = os.path.join(config['raw_images_path'], filename)
            save_path = os.path.join(MASKS_PATH, filename)
            generate_mask(image_path, save_path)
        else:
            logging.warning(f"Skipping non-image file: {filename}")

if __name__ == "__main__":
    main() 