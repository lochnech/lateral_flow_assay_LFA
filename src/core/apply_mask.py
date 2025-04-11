import cv2
import os
import numpy as np

def apply_mask_to_image(image_path, mask_path, output_path):
    # Read original image and mask
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return
        
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Failed to load mask: {mask_path}")
        return

    # Create a colored overlay for the mask (e.g., red)
    overlay = image.copy()
    overlay[mask > 0] = [0, 0, 255]  # Red color where mask is white

    # Blend the original image with the overlay
    alpha = 0.3  # Transparency factor
    output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Save the result
    cv2.imwrite(output_path, output)
    print(f"Saved masked image to: {output_path}")

def main():
    # Directory paths
    input_dir = "./data/processed/padded_images/"
    mask_dir = "./data/processed/masks/"
    output_dir = "./data/processed/masked_images/"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Debug: Print available files
    print(f"Input images: {os.listdir(input_dir)}")
    print(f"Available masks: {os.listdir(mask_dir)}")

    # Process each image
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            # Try both lowercase and uppercase extensions
            base_name = os.path.splitext(filename)[0]
            possible_mask_names = [
                f"{base_name}.png",
                f"{base_name}.PNG",
                f"{base_name}.jpg",
                f"{base_name}.JPG",
                f"{base_name}.jpeg",
                f"{base_name}.JPEG"
            ]
            
            mask_found = False
            for mask_name in possible_mask_names:
                mask_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(mask_path):
                    mask_found = True
                    output_path = os.path.join(output_dir, f"masked_{filename}")
                    print(f"Processing {filename} with mask {mask_name}")
                    apply_mask_to_image(image_path, mask_path, output_path)
                    break
            
            if not mask_found:
                print(f"Warning: No mask found for {filename}")

if __name__ == "__main__":
    main()
