import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any
import cv2
import numpy as np

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from lfa.utils.image_utils import load_image, save_image, overlay_mask
from lfa.utils.logging_utils import setup_logging

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def process_image(image_path: str, 
                 mask_path: str, 
                 output_path: str,
                 config: Dict[str, Any],
                 logger) -> None:
    """Process a single image and its mask
    
    Args:
        image_path (str): Path to the input image
        mask_path (str): Path to the mask image
        output_path (str): Path to save the output image
        config (Dict[str, Any]): Configuration dictionary
        logger: Logger instance
    """
    try:
        # Load images
        image = load_image(image_path)
        mask = cv2.imread(mask_path, 0)
        
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return
        
        if mask is None:
            logger.error(f"Could not load mask: {mask_path}")
            return
        
        # Create overlay
        overlay = overlay_mask(
            image,
            mask,
            color=tuple(config['postprocessing']['overlay_color']),
            alpha=config['postprocessing']['overlay_alpha']
        )
        
        # Save result
        save_image(overlay, output_path)
        logger.info(f"Applied mask for {image_path} -> {output_path}")
        
        # Print statistics
        mask_pixels = np.count_nonzero(mask)
        total_pixels = mask.size
        coverage = mask_pixels / total_pixels * 100
        logger.info(f"Mask coverage: {mask_pixels} pixels ({coverage:.2f}%)")
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")

def main():
    # Load configuration
    config = load_config('lfa/config/default_config.yaml')
    
    # Create output directory
    output_dir = Path(config['data']['result_dir']) / 'overlays'
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(config['data']['log_dir'], name='mask_application')
    
    # Process all images in the input directory
    image_dir = Path(config['data']['test_image_dir'])
    mask_dir = Path(config['data']['result_dir'])
    
    for image_path in image_dir.glob('*.png'):
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            logger.warning(f"No mask found for {image_path}")
            continue
            
        output_path = output_dir / image_path.name
        process_image(str(image_path), str(mask_path), str(output_path), config, logger)
    
    logger.info("Mask application complete")

if __name__ == "__main__":
    main()
