import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
from segmentation_ROI import UNET
from utils import load_checkpoint

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
LOAD_CHECKPOINT_PATH = "./models/model_checkpoint.pth.tar"
DATA_PATH = "./data"

# Define transformation for input image
#transform = A.Compose([
 #   A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
 #   A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
 #   ToTensorV2(),
#])
# Define transformation for input image
transform = A.Compose([
    A.Rotate(limit=(90, 90), always_apply=True),  # Rotate exactly 90 degrees
    A.PadIfNeeded(min_height=IMAGE_HEIGHT, min_width=IMAGE_WIDTH, border_mode=1),
    #A.PadIfNeeded(min_height=IMAGE_HEIGHT, min_width=IMAGE_WIDTH, border_mode=0, value=(0, 0, 0)),  # Fills with black
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),  # Ensures final size
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

# Load model
model = UNET(in_channels=3, out_channels=1).to(DEVICE)
load_checkpoint(torch.load(LOAD_CHECKPOINT_PATH, map_location=DEVICE), model)
model.eval()

# Function to generate mask
def generate_mask(image_path, save_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    input_tensor = augmented["image"].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.sigmoid(output).cpu().numpy().squeeze()
        mask = (mask > 0.5).astype(np.uint8) * 255  # Thresholding for binary mask
    
    cv2.imwrite(save_path, mask)
    print(f"Mask saved to {save_path}")

# Example usage
input_folder = "./data/test_images/"
output_folder = "./data/result_images/"
for filename in os.listdir(input_folder):
    image_path = os.path.join(input_folder, filename)
    save_path = os.path.join(output_folder, filename)
    generate_mask(image_path, save_path)