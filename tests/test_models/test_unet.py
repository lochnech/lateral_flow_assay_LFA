import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from lfa.models.unet import UNet, DoubleConv, Down, Up, OutConv

@pytest.fixture
def sample_input():
    """Create a sample input tensor"""
    return torch.randn(1, 3, 256, 256)  # Batch size 1, 3 channels, 256x256 image

@pytest.fixture
def sample_mask():
    """Create a sample target mask"""
    return torch.randint(0, 2, (1, 1, 256, 256)).float()  # Binary mask

def test_double_conv():
    """Test DoubleConv block"""
    # Test with different input/output channels
    in_channels = 3
    out_channels = 64
    double_conv = DoubleConv(in_channels, out_channels)
    
    # Create input tensor
    x = torch.randn(1, in_channels, 256, 256)
    
    # Forward pass
    output = double_conv(x)
    
    # Check output properties
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, out_channels, 256, 256)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_down_block():
    """Test Down block"""
    # Test with different input/output channels
    in_channels = 3
    out_channels = 64
    down = Down(in_channels, out_channels)
    
    # Create input tensor
    x = torch.randn(1, in_channels, 256, 256)
    
    # Forward pass
    output, skip = down(x)
    
    # Check output properties
    assert isinstance(output, torch.Tensor)
    assert isinstance(skip, torch.Tensor)
    assert output.shape == (1, out_channels, 128, 128)  # Downsampled
    assert skip.shape == (1, out_channels, 256, 256)  # Skip connection
    assert not torch.isnan(output).any()
    assert not torch.isnan(skip).any()

def test_up_block():
    """Test Up block"""
    # Test with different input/output channels
    in_channels = 64
    out_channels = 32
    up = Up(in_channels, out_channels)
    
    # Create input tensors
    x = torch.randn(1, in_channels, 128, 128)
    skip = torch.randn(1, in_channels, 256, 256)
    
    # Forward pass
    output = up(x, skip)
    
    # Check output properties
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, out_channels, 256, 256)  # Upsampled
    assert not torch.isnan(output).any()

def test_out_conv():
    """Test OutConv block"""
    # Test with different input/output channels
    in_channels = 64
    out_channels = 1
    out_conv = OutConv(in_channels, out_channels)
    
    # Create input tensor
    x = torch.randn(1, in_channels, 256, 256)
    
    # Forward pass
    output = out_conv(x)
    
    # Check output properties
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, out_channels, 256, 256)
    assert not torch.isnan(output).any()

def test_unet_forward(sample_input):
    """Test UNet forward pass"""
    # Create model
    model = UNet(n_channels=3, n_classes=1)
    
    # Forward pass
    output = model(sample_input)
    
    # Check output properties
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 1, 256, 256)  # Same spatial dimensions as input
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_unet_with_different_input_sizes():
    """Test UNet with different input sizes"""
    model = UNet(n_channels=3, n_classes=1)
    
    # Test with different input sizes
    sizes = [(128, 128), (256, 256), (512, 512)]
    for h, w in sizes:
        x = torch.randn(1, 3, h, w)
        output = model(x)
        assert output.shape == (1, 1, h, w)

def test_unet_with_different_batch_sizes():
    """Test UNet with different batch sizes"""
    model = UNet(n_channels=3, n_classes=1)
    
    # Test with different batch sizes
    batch_sizes = [1, 2, 4, 8]
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 3, 256, 256)
        output = model(x)
        assert output.shape == (batch_size, 1, 256, 256)

def test_unet_with_different_channels():
    """Test UNet with different input/output channels"""
    # Test different input channels
    for in_channels in [1, 3, 4]:
        model = UNet(n_channels=in_channels, n_classes=1)
        x = torch.randn(1, in_channels, 256, 256)
        output = model(x)
        assert output.shape == (1, 1, 256, 256)
    
    # Test different output channels
    for out_channels in [1, 2, 3]:
        model = UNet(n_channels=3, n_classes=out_channels)
        x = torch.randn(1, 3, 256, 256)
        output = model(x)
        assert output.shape == (1, out_channels, 256, 256)

def test_unet_gradients(sample_input, sample_mask):
    """Test UNet gradient computation"""
    model = UNet(n_channels=3, n_classes=1)
    
    # Forward pass
    output = model(sample_input)
    
    # Compute loss
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(output, sample_mask)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    for name, param in model.named_parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
        assert not torch.isinf(param.grad).any()

def test_unet_device_moving():
    """Test moving UNet between devices"""
    model = UNet(n_channels=3, n_classes=1)
    
    # Test moving to CPU (if not already there)
    model = model.cpu()
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    assert output.device.type == 'cpu'
    
    # Test moving to CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        output = model(x)
        assert output.device.type == 'cuda'
