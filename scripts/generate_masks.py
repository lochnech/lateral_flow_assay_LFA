import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
import cv2

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from lfa.models.unet import load_unet_model
from lfa.models.unetpp import load_unetpp_model
from lfa.utils.image_utils import preprocess_image, postprocess_mask, save_image
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

def create_model(config: Dict[str, Any]) -> torch.nn.Module:
    """Create and load model based on configuration
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        torch.nn.Module: Loaded model
    """
    model_config = config['model']
    inference_config = config['inference']
    
    if model_config['name'] == 'unet':
        model = load_unet_model(
            checkpoint_path=inference_config['checkpoint_path'],
            device=inference_config['device']
        )
    elif model_config['name'] == 'unetpp':
        model = load_unetpp_model(
            checkpoint_path=inference_config['checkpoint_path'],
            device=inference_config['device']
        )
    else:
        raise ValueError(f"Unknown model name: {model_config['name']}")
    
    return model

def process_image(model: torch.nn.Module, 
                 image_path: str, 
                 output_path: str,
                 config: Dict[str, Any],
                 logger) -> None:
    """Process a single image and save the mask
    
    Args:
        model (torch.nn.Module): Model to use for prediction
        image_path (str): Path to the input image
        output_path (str): Path to save the output mask
        config (Dict[str, Any]): Configuration dictionary
        logger: Logger instance
    """
    try:
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_image(image, size=tuple(config['augmentation']['image_size']))
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(preprocessed).unsqueeze(0)
        tensor = tensor.to(config['inference']['device'])
        
        # Generate mask
        with torch.no_grad():
            output = model(tensor)
            mask = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Postprocess mask
        mask = postprocess_mask(
            mask,
            threshold=config['inference']['threshold']
        )
        
        # Save mask
        save_image(mask, output_path)
        logger.info(f"Generated mask for {image_path} -> {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")

def main():
    # Load configuration
    config = load_config('lfa/config/default_config.yaml')
    
    # Create output directory
    os.makedirs(config['data']['result_dir'], exist_ok=True)
    
    # Setup logging
    logger = setup_logging(config['data']['log_dir'], name='mask_generation')
    
    # Create and load model
    model = create_model(config)
    model.eval()
    
    # Process all images in the input directory
    input_dir = Path(config['data']['test_image_dir'])
    output_dir = Path(config['data']['result_dir'])
    
    for image_path in input_dir.glob('*.png'):
        output_path = output_dir / image_path.name
        process_image(model, str(image_path), str(output_path), config, logger)
    
    logger.info("Mask generation complete")

if __name__ == "__main__":
    main()
