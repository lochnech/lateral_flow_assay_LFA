import cv2
import os
import numpy as np

def apply_mask_to_image(image_path, mask_path, output_path):
    """Apply mask to image and save the result"""
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return
    
    # Load the mask
    mask = cv2.imread(mask_path, 0)  # Load as grayscale
    if mask is None:
        print(f"Error: Failed to load mask: {mask_path}")
        return
    
    # Convert image to RGB for better visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a copy of the image for overlay
    overlay = image_rgb.copy()
    
    # Resize mask to match image dimensions if they don't match
    if mask.shape != image_rgb.shape[:2]:
        print(f"Resizing mask from {mask.shape} to {image_rgb.shape[:2]}")
        mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]), 
                         interpolation=cv2.INTER_LINEAR)
    
    # Create a red overlay (in RGB format)
    red_overlay = np.zeros_like(image_rgb)
    red_overlay[:] = [255, 0, 0]  # Red color in RGB
    
    # Create alpha channel from mask
    alpha = mask.astype(float) / 255.0
    alpha = np.stack([alpha] * 3, axis=-1)
    
    # Blend the original image with the red overlay
    overlay = (1 - alpha) * overlay + alpha * red_overlay
    overlay = overlay.astype(np.uint8)
    
    # Convert back to BGR for saving
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the result
    cv2.imwrite(output_path, overlay_bgr)
    print(f"Saved masked image to: {output_path}")
    
    # Print some statistics
    print(f"Mask coverage: {np.count_nonzero(mask)} pixels ({np.count_nonzero(mask)/mask.size*100:.2f}%)")
    print(f"Red pixels in overlay: {np.sum(np.all(overlay == [255,0,0], axis=2))} pixels")

def main():
    # Directory containing the images
    image_dir = "./data/test_images/"
    mask_dir = "./data/result_images/"
    output_dir = "./data/masked_images/"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            # Construct paths
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Apply mask and save result
            apply_mask_to_image(image_path, mask_path, output_path)

if __name__ == "__main__":
    main()
