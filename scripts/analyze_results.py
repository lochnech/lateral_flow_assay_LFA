import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from lfa.utils.image_utils import load_image, calculate_iou
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

def analyze_mask(mask: np.ndarray) -> Dict[str, float]:
    """Analyze a single mask and return statistics
    
    Args:
        mask (np.ndarray): Binary mask image
        
    Returns:
        Dict[str, float]: Dictionary of mask statistics
    """
    # Ensure mask is binary
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    
    # Calculate basic statistics
    total_pixels = mask.size
    mask_pixels = np.count_nonzero(mask)
    coverage = mask_pixels / total_pixels * 100
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask)
    
    # Calculate component statistics
    component_sizes = []
    for label in range(1, num_labels):
        component_sizes.append(np.sum(labels == label))
    
    if component_sizes:
        avg_size = np.mean(component_sizes)
        max_size = np.max(component_sizes)
        min_size = np.min(component_sizes)
    else:
        avg_size = max_size = min_size = 0
    
    return {
        'coverage_percentage': coverage,
        'total_pixels': total_pixels,
        'mask_pixels': mask_pixels,
        'num_components': num_labels - 1,
        'avg_component_size': avg_size,
        'max_component_size': max_size,
        'min_component_size': min_size
    }

def compare_masks(predicted_mask: np.ndarray, 
                 ground_truth_mask: np.ndarray) -> Dict[str, float]:
    """Compare predicted mask with ground truth mask
    
    Args:
        predicted_mask (np.ndarray): Predicted binary mask
        ground_truth_mask (np.ndarray): Ground truth binary mask
        
    Returns:
        Dict[str, float]: Dictionary of comparison metrics
    """
    # Ensure masks are binary
    if predicted_mask.max() > 1:
        predicted_mask = (predicted_mask > 127).astype(np.uint8)
    if ground_truth_mask.max() > 1:
        ground_truth_mask = (ground_truth_mask > 127).astype(np.uint8)
    
    # Calculate IoU
    iou = calculate_iou(predicted_mask, ground_truth_mask)
    
    # Calculate precision and recall
    true_positives = np.sum(np.logical_and(predicted_mask, ground_truth_mask))
    false_positives = np.sum(np.logical_and(predicted_mask, 1 - ground_truth_mask))
    false_negatives = np.sum(np.logical_and(1 - predicted_mask, ground_truth_mask))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    }

def generate_plots(stats_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate analysis plots
    
    Args:
        stats_df (pd.DataFrame): DataFrame containing statistics
        output_dir (Path): Directory to save plots
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot coverage distribution
    plt.figure(figsize=(10, 6))
    plt.hist(stats_df['coverage_percentage'], bins=20)
    plt.title('Mask Coverage Distribution')
    plt.xlabel('Coverage Percentage')
    plt.ylabel('Number of Images')
    plt.savefig(output_dir / 'coverage_distribution.png')
    plt.close()
    
    # Plot component size distribution
    plt.figure(figsize=(10, 6))
    plt.hist(stats_df['avg_component_size'], bins=20)
    plt.title('Average Component Size Distribution')
    plt.xlabel('Component Size (pixels)')
    plt.ylabel('Number of Images')
    plt.savefig(output_dir / 'component_size_distribution.png')
    plt.close()
    
    # Plot IoU distribution if ground truth masks are available
    if 'iou' in stats_df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(stats_df['iou'], bins=20)
        plt.title('IoU Distribution')
        plt.xlabel('IoU Score')
        plt.ylabel('Number of Images')
        plt.savefig(output_dir / 'iou_distribution.png')
        plt.close()

def main():
    # Load configuration
    config = load_config('lfa/config/default_config.yaml')
    
    # Setup logging
    logger = setup_logging(config['data']['log_dir'], name='analysis')
    
    # Create output directories
    output_dir = Path(config['data']['result_dir']) / 'analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize statistics list
    stats_list = []
    
    # Process all masks in the result directory
    mask_dir = Path(config['data']['result_dir'])
    ground_truth_dir = Path(config['data']['test_mask_dir'])
    
    for mask_path in mask_dir.glob('*.png'):
        try:
            # Load mask
            mask = cv2.imread(str(mask_path), 0)
            if mask is None:
                logger.error(f"Could not load mask: {mask_path}")
                continue
            
            # Analyze mask
            mask_stats = analyze_mask(mask)
            mask_stats['filename'] = mask_path.name
            
            # Compare with ground truth if available
            gt_path = ground_truth_dir / mask_path.name
            if gt_path.exists():
                gt_mask = cv2.imread(str(gt_path), 0)
                if gt_mask is not None:
                    comparison_stats = compare_masks(mask, gt_mask)
                    mask_stats.update(comparison_stats)
            
            stats_list.append(mask_stats)
            logger.info(f"Analyzed {mask_path.name}")
            
        except Exception as e:
            logger.error(f"Error analyzing {mask_path}: {str(e)}")
    
    # Create DataFrame and save statistics
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(output_dir / 'mask_statistics.csv', index=False)
    
    # Generate plots
    generate_plots(stats_df, output_dir)
    
    # Print summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"Total images analyzed: {len(stats_df)}")
    logger.info(f"Average coverage: {stats_df['coverage_percentage'].mean():.2f}%")
    logger.info(f"Average components per image: {stats_df['num_components'].mean():.2f}")
    
    if 'iou' in stats_df.columns:
        logger.info(f"Average IoU: {stats_df['iou'].mean():.4f}")
        logger.info(f"Average F1 Score: {stats_df['f1_score'].mean():.4f}")
    
    logger.info(f"Analysis complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
