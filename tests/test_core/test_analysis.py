import os
import sys
import pytest
import numpy as np
from pathlib import Path
import cv2
import pandas as pd

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from lfa.core.analysis import (
    analyze_mask,
    compare_masks,
    generate_plots
)

@pytest.fixture
def sample_mask():
    """Create a sample test mask"""
    # Create a binary mask with two connected components
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[64:96, 64:96] = 255  # First component
    mask[160:192, 160:192] = 255  # Second component
    return mask

@pytest.fixture
def ground_truth_mask():
    """Create a ground truth mask for testing"""
    # Create a binary mask with some overlap with sample_mask
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[64:96, 64:96] = 255  # Perfect overlap with first component
    mask[160:180, 160:180] = 255  # Partial overlap with second component
    return mask

def test_analyze_mask(sample_mask):
    """Test mask analysis"""
    stats = analyze_mask(sample_mask)
    
    # Check statistics
    assert isinstance(stats, dict)
    assert 'coverage_percentage' in stats
    assert 'total_pixels' in stats
    assert 'mask_pixels' in stats
    assert 'num_components' in stats
    assert 'avg_component_size' in stats
    assert 'max_component_size' in stats
    assert 'min_component_size' in stats
    
    # Check specific values
    assert stats['total_pixels'] == 256 * 256
    assert stats['mask_pixels'] == 2 * 32 * 32  # Two 32x32 squares
    assert stats['num_components'] == 2
    assert stats['max_component_size'] == 32 * 32
    assert stats['min_component_size'] == 32 * 32
    assert stats['avg_component_size'] == 32 * 32

def test_compare_masks(sample_mask, ground_truth_mask):
    """Test mask comparison"""
    metrics = compare_masks(sample_mask, ground_truth_mask)
    
    # Check metrics
    assert isinstance(metrics, dict)
    assert 'iou' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    
    # Check metric ranges
    assert 0 <= metrics['iou'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1_score'] <= 1
    
    # Check specific values for our test case
    # First component has perfect overlap (32x32)
    # Second component has partial overlap (20x20)
    total_overlap = 32*32 + 20*20
    total_predicted = 2 * 32*32
    total_ground_truth = 32*32 + 20*20
    
    expected_iou = total_overlap / (total_predicted + total_ground_truth - total_overlap)
    assert abs(metrics['iou'] - expected_iou) < 1e-6

def test_generate_plots(tmp_path):
    """Test plot generation"""
    # Create sample statistics DataFrame
    stats_df = pd.DataFrame({
        'coverage_percentage': [10, 20, 30, 40, 50],
        'avg_component_size': [100, 200, 300, 400, 500],
        'iou': [0.1, 0.2, 0.3, 0.4, 0.5]
    })
    
    # Generate plots
    generate_plots(stats_df, tmp_path)
    
    # Check that plots were created
    assert (tmp_path / 'coverage_distribution.png').exists()
    assert (tmp_path / 'component_size_distribution.png').exists()
    assert (tmp_path / 'iou_distribution.png').exists()

def test_analysis_pipeline(tmp_path, sample_mask, ground_truth_mask):
    """Test the complete analysis pipeline"""
    # Create temporary directories
    mask_dir = tmp_path / "masks"
    gt_dir = tmp_path / "ground_truth"
    output_dir = tmp_path / "analysis"
    mask_dir.mkdir()
    gt_dir.mkdir()
    output_dir.mkdir()
    
    # Save sample masks
    cv2.imwrite(str(mask_dir / "test.png"), sample_mask)
    cv2.imwrite(str(gt_dir / "test.png"), ground_truth_mask)
    
    # Initialize statistics list
    stats_list = []
    
    # Process all masks
    for mask_path in mask_dir.glob("*.png"):
        # Load mask
        mask = cv2.imread(str(mask_path), 0)
        if mask is None:
            continue
        
        # Analyze mask
        mask_stats = analyze_mask(mask)
        mask_stats['filename'] = mask_path.name
        
        # Compare with ground truth if available
        gt_path = gt_dir / mask_path.name
        if gt_path.exists():
            gt_mask = cv2.imread(str(gt_path), 0)
            if gt_mask is not None:
                comparison_stats = compare_masks(mask, gt_mask)
                mask_stats.update(comparison_stats)
        
        stats_list.append(mask_stats)
    
    # Create DataFrame and save statistics
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(output_dir / 'mask_statistics.csv', index=False)
    
    # Generate plots
    generate_plots(stats_df, output_dir)
    
    # Verify output
    assert (output_dir / 'mask_statistics.csv').exists()
    assert (output_dir / 'coverage_distribution.png').exists()
    assert (output_dir / 'component_size_distribution.png').exists()
    assert (output_dir / 'iou_distribution.png').exists()
    
    # Verify statistics
    assert len(stats_df) == 1
    assert 'filename' in stats_df.columns
    assert 'coverage_percentage' in stats_df.columns
    assert 'iou' in stats_df.columns
