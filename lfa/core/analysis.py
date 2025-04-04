import cv2
import numpy as np
import os
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_mask_metrics(mask: np.ndarray) -> Dict[str, float]:
    """Calculate various metrics for a mask
    
    Args:
        mask (np.ndarray): Binary mask image
        
    Returns:
        Dict[str, float]: Dictionary containing mask metrics
    """
    # Convert to binary if not already
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    
    # Calculate basic metrics
    total_pixels = mask.size
    mask_pixels = np.count_nonzero(mask)
    coverage = mask_pixels / total_pixels * 100
    
    # Calculate connected components
    num_labels, labels = cv2.connectedComponents(mask)
    
    # Calculate component sizes
    component_sizes = []
    for label in range(1, num_labels):
        component_sizes.append(np.sum(labels == label))
    
    # Calculate statistics
    if component_sizes:
        avg_size = np.mean(component_sizes)
        max_size = np.max(component_sizes)
        min_size = np.min(component_sizes)
        num_components = len(component_sizes)
    else:
        avg_size = max_size = min_size = num_components = 0
    
    return {
        'coverage_percentage': coverage,
        'total_pixels': total_pixels,
        'mask_pixels': mask_pixels,
        'num_components': num_components,
        'avg_component_size': avg_size,
        'max_component_size': max_size,
        'min_component_size': min_size
    }

def analyze_mask_directory(mask_dir: str) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Analyze all masks in a directory
    
    Args:
        mask_dir (str): Directory containing mask images
        
    Returns:
        Tuple[Dict[str, float], List[Dict[str, float]]]: Overall statistics and per-image statistics
    """
    mask_dir = Path(mask_dir)
    per_image_stats = []
    
    # Process each mask
    for mask_path in mask_dir.glob('*.png'):
        mask = cv2.imread(str(mask_path), 0)
        if mask is None:
            print(f"Warning: Could not read mask {mask_path}")
            continue
            
        stats = calculate_mask_metrics(mask)
        stats['filename'] = mask_path.name
        per_image_stats.append(stats)
    
    # Calculate overall statistics
    if per_image_stats:
        overall_stats = {
            'avg_coverage': np.mean([s['coverage_percentage'] for s in per_image_stats]),
            'std_coverage': np.std([s['coverage_percentage'] for s in per_image_stats]),
            'avg_components': np.mean([s['num_components'] for s in per_image_stats]),
            'total_images': len(per_image_stats)
        }
    else:
        overall_stats = {
            'avg_coverage': 0,
            'std_coverage': 0,
            'avg_components': 0,
            'total_images': 0
        }
    
    return overall_stats, per_image_stats

def plot_mask_analysis(overall_stats: Dict[str, float], 
                      per_image_stats: List[Dict[str, float]],
                      output_dir: str):
    """Generate plots for mask analysis
    
    Args:
        overall_stats (Dict[str, float]): Overall statistics
        per_image_stats (List[Dict[str, float]]): Per-image statistics
        output_dir (str): Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot coverage distribution
    plt.figure(figsize=(10, 6))
    coverages = [s['coverage_percentage'] for s in per_image_stats]
    plt.hist(coverages, bins=20)
    plt.title('Mask Coverage Distribution')
    plt.xlabel('Coverage Percentage')
    plt.ylabel('Number of Images')
    plt.savefig(output_dir / 'coverage_distribution.png')
    plt.close()
    
    # Plot component size distribution
    plt.figure(figsize=(10, 6))
    component_sizes = [s['avg_component_size'] for s in per_image_stats]
    plt.hist(component_sizes, bins=20)
    plt.title('Average Component Size Distribution')
    plt.xlabel('Component Size (pixels)')
    plt.ylabel('Number of Images')
    plt.savefig(output_dir / 'component_size_distribution.png')
    plt.close()
    
    # Save statistics to text file
    with open(output_dir / 'statistics.txt', 'w') as f:
        f.write("Overall Statistics:\n")
        for key, value in overall_stats.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nPer-Image Statistics:\n")
        for stats in per_image_stats:
            f.write(f"\nImage: {stats['filename']}\n")
            for key, value in stats.items():
                if key != 'filename':
                    f.write(f"{key}: {value}\n")

def main():
    """Main function to analyze masks in a directory"""
    mask_dir = "./data/result_images/"
    output_dir = "./data/analysis_results/"
    
    overall_stats, per_image_stats = analyze_mask_directory(mask_dir)
    plot_mask_analysis(overall_stats, per_image_stats, output_dir)
    
    print("Analysis complete. Results saved to:", output_dir)

if __name__ == "__main__":
    main()
