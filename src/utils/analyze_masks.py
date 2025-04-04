import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_mask(test_image_path, mask_image_path, masked_image_path, output_path):
    """Analyze a single mask and save the analysis image"""
    # Load images
    img = cv2.imread(test_image_path)
    mask = cv2.imread(mask_image_path, 0)
    masked = cv2.imread(masked_image_path)
    
    if img is None or mask is None or masked is None:
        print(f"Error: Failed to load one or more images for {test_image_path}")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12,12))
    
    # Plot original image
    axes[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title('Original Image')
    
    # Plot mask
    axes[0,1].imshow(mask, cmap='gray')
    axes[0,1].set_title('Mask')
    
    # Plot masked image
    axes[1,0].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
    axes[1,0].set_title('Masked Image')
    
    # Plot mask applied to original
    axes[1,1].imshow(cv2.bitwise_and(img, img, mask=mask))
    axes[1,1].set_title('Mask Applied')
    
    # Add statistics
    stats_text = f"Mask Coverage: {np.count_nonzero(mask)/mask.size*100:.2f}%"
    fig.text(0.5, 0.01, stats_text, ha='center', fontsize=12)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the analysis image
    plt.savefig(output_path)
    plt.close()
    print(f"Saved analysis for {os.path.basename(test_image_path)} to {output_path}")

def main():
    # Directory paths
    test_dir = "./data/test_images/"
    masks_dir = "./data/masks/"
    masked_dir = "./data/masked_images/"
    analysis_dir = "./data/analysis/"
    
    # Create analysis directory if it doesn't exist
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Process each test image
    for filename in os.listdir(test_dir):
        if filename.endswith(".png"):
            # Construct paths
            test_path = os.path.join(test_dir, filename)
            mask_path = os.path.join(masks_dir, filename)
            masked_path = os.path.join(masked_dir, filename)
            analysis_path = os.path.join(analysis_dir, f"analysis_{filename}")
            
            # Analyze the mask
            analyze_mask(test_path, mask_path, masked_path, analysis_path)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 