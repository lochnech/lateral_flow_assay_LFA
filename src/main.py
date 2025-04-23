import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

RAW_IMAGES_PATH = config['raw_images_path']
MASKS_PATH = config['masks_path']

# Loop through all images in the directory
for filename in os.listdir(RAW_IMAGES_PATH):

    # Load the image
    image_path = os.path.join(RAW_IMAGES_PATH, filename)
    image = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    # plt.show()

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper red color ranges in HSV
    lower_red1 = np.array([0, 50, 50])  # Adjusted for more flexibility
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red regions
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    # Combine both masks
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Debugging step: Show the red mask
    plt.imshow(red_mask, cmap='gray')
    plt.title('Red Mask')
    plt.axis('off')
    # plt.show()

    # Apply the mask to the original image
    red_detected = cv2.bitwise_and(image, image, mask=red_mask)

    # Detect edges in the masked red areas
    edges = cv2.Canny(red_mask, 50, 150)  # Detect edges on red areas

    # Debugging: Show edges to check if they are detected
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')
    # plt.show()

    # Detect lines using Hough Transform with optimized parameters
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=20)

    # Draw detected lines on the original image
    image_with_lines = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Draw green lines

    # Show the final image with detected red lines marked in green
    plt.imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))
    plt.title('Image with Red Lines Marked in Green')
    plt.axis('off')
    plt.show()

    # # Save the final image with detected red lines marked in green
    # output_path = os.path.join(image_directory+"/lines", f'{filename}_with_lines.jpg')
    # cv2.imwrite(output_path, image_with_lines)

    # # Print a message to confirm the image has been saved
    # print(f"Image with lines saved to {output_path}")
