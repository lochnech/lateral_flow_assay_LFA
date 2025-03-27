# UNET architecture for segmentation training with mask and predicting the mask
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling path)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder (Upsampling path)
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        # Bottleneck and final convolution
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder path with skip connections
        for index in range(0, len(self.ups), 2):
            x = self.ups[index](x)
            skip_connection = skip_connections[index // 2]
            
            # Handle size mismatches in skip connections
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            # Concatenate skip connection and continue
            concatenate_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[index + 1](concatenate_skip)
        
        return self.final_conv(x)


def test():
    """Test the UNET model with different configurations"""
    # Test configurations
    configs = [
        {"in_channels": 3, "out_channels": 1, "features": [64, 128, 256, 512]},
        {"in_channels": 1, "out_channels": 1, "features": [32, 64, 128, 256]},
    ]
    
    for config in configs:
        print(f"\nTesting configuration: {config}")
        
        # Create model
        model = UNET(**config)
        
        # Create input tensor
        batch_size = 1
        input_size = 512  # Using standard size
        x = torch.randn((batch_size, config["in_channels"], input_size, input_size))
        
        # Test forward pass
        try:
            preds = model(x)
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {preds.shape}")
            
            # Verify output shape
            expected_shape = (batch_size, config["out_channels"], input_size, input_size)
            assert preds.shape == expected_shape, f"Expected shape {expected_shape}, got {preds.shape}"
            print("✓ Shape verification passed")
            
            # Verify output values
            assert torch.isfinite(preds).all(), "Output contains invalid values"
            print("✓ Output values verification passed")
            
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")


if __name__ == "__main__":
    test()
