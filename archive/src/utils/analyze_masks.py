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
    
    # Convert to grayscale for intensity analysis
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    
    # Calculate intensity profile
    intensity_profile = np.mean(gray, axis=0)
    
    # Create figure with more subplots
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(3, 2)
    
    # Plot original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    
    # Plot mask
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Mask')
    
    # Plot masked image
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
    ax3.set_title('Masked Image')
    
    # Plot mask applied to original
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(cv2.bitwise_and(img, img, mask=mask))
    ax4.set_title('Mask Applied')
    
    # Plot intensity heatmap
    ax5 = fig.add_subplot(gs[2, 0])
    heatmap = ax5.imshow(gray, cmap='hot', aspect='auto')
    plt.colorbar(heatmap, ax=ax5)
    ax5.set_title('Intensity Heatmap')
    
    # Plot intensity profile
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(intensity_profile)
    ax6.set_title('Intensity Profile')
    ax6.set_xlabel('Pixel Position')
    ax6.set_ylabel('Intensity')
    
    # Add statistics
    stats_text = f"Mask Coverage: {np.count_nonzero(mask)/mask.size*100:.2f}%\n"
    stats_text += f"Max Intensity: {np.max(gray)}\n"
    stats_text += f"Min Intensity: {np.min(gray)}\n"
    stats_text += f"Mean Intensity: {np.mean(gray):.2f}"
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