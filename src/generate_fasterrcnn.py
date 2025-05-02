import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
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
DETECTIONS_PATH = config['fasterrcnn']['detections_path']
VISUALIZED_DETECTIONS_PATH = config['fasterrcnn']['visualized_detections_path']

def setup_logging():
    """Setup logging configuration"""
    os.makedirs(LOGS_PATH, exist_ok=True)
    logging.basicConfig(
        filename=f'{LOGS_PATH}/fasterrcnn_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_model():
    """Load and configure the Faster R-CNN model"""
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Modify the model to handle our specific case
    # Set a higher confidence threshold for detection
    model.roi_heads.score_thresh = 0.2  # Higher threshold for detection
    model.roi_heads.nms_thresh = 0.3    # Higher NMS threshold to reduce overlapping detections
    
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image_path):
    """Load and preprocess an image for Faster R-CNN"""
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

def process_detections(predictions, image_shape):
    """Process model predictions into bounding boxes and scores"""
    detections = []
    
    if len(predictions[0]['boxes']) > 0:
        for box, score, label in zip(predictions[0]['boxes'], predictions[0]['scores'], predictions[0]['labels']):
            if score > 0.2:  # Higher threshold for detection
                box = box.cpu().numpy().astype(int)
                detections.append({
                    'box': box,
                    'score': score.item(),
                    'label': label.item()
                })
    
    return detections

def save_debug_visualization(image, detections, save_path):
    """Save debug visualization of the detections"""
    debug_image = image.copy()
    
    # Draw bounding boxes and scores
    for detection in detections:
        box = detection['box']
        score = detection['score']
        label = detection['label']
        
        # Draw box
        cv2.rectangle(debug_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # Add label and score
        label_text = f"Class {label}: {score:.2f}"
        cv2.putText(debug_image, label_text, (box[0], box[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save debug image
    cv2.imwrite(save_path, debug_image)

def generate_detections(image_path, save_path):
    """Generate detections using Faster R-CNN model"""
    # Load and preprocess image
    image_tensor, image_rgb = preprocess_image(image_path)
    if image_tensor is None:
        return
    
    # Get model predictions
    with torch.no_grad():
        predictions = model([image_tensor.to(DEVICE)])
    
    # Process predictions into detections
    detections = process_detections(predictions, image_rgb.shape)
    
    # Save detections
    os.makedirs(DETECTIONS_PATH, exist_ok=True)
    detection_file = os.path.join(DETECTIONS_PATH, f"{os.path.splitext(os.path.basename(image_path))[0]}_detections.txt")
    
    with open(detection_file, 'w') as f:
        for detection in detections:
            box = detection['box']
            score = detection['score']
            label = detection['label']
            f.write(f"Class {label}: Score {score:.3f}, Box {box}\n")
    
    # Save debug visualization
    os.makedirs(VISUALIZED_DETECTIONS_PATH, exist_ok=True)
    debug_path = os.path.join(VISUALIZED_DETECTIONS_PATH, f"debug_{os.path.basename(image_path)}")
    save_debug_visualization(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), detections, debug_path)
    
    # Log results
    logging.info(f"Processed image: {os.path.basename(image_path)}")
    if detections:
        logging.info(f"Number of detections: {len(detections)}")
        for detection in detections:
            logging.info(f"Detection: Class {detection['label']}, Score {detection['score']:.3f}")

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
            save_path = os.path.join(DETECTIONS_PATH, filename)
            generate_detections(image_path, save_path)
        else:
            logging.warning(f"Skipping non-image file: {filename}")

if __name__ == "__main__":
    main() 