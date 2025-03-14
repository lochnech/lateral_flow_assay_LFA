import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNETpp(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder (Downsampling path)
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        
        # Nested Decoder blocks
        self.dec1_0 = ConvBlock(512 + 256, 256)
        self.dec2_0 = ConvBlock(256 + 128, 128)
        self.dec3_0 = ConvBlock(128 + 64, 64)
        
        self.dec2_1 = ConvBlock(256 + 128 + 128, 128)
        self.dec3_1 = ConvBlock(128 + 64 + 64, 64)
        
        self.dec3_2 = ConvBlock(128 + 64 + 64 + 64, 64)
        
        # Final outputs for deep supervision
        self.final1 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.final2 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.final3 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.final4 = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        
        # Decoder with nested dense connections
        # First decoder path (L=4)
        d1_0 = self.dec1_0(torch.cat([self.up(x4), x3], dim=1))
        d2_0 = self.dec2_0(torch.cat([self.up(d1_0), x2], dim=1))
        d3_0 = self.dec3_0(torch.cat([self.up(d2_0), x1], dim=1))
        out1 = self.final1(d3_0)
        
        # Second decoder path (L=3)
        d2_1 = self.dec2_1(torch.cat([self.up(d1_0), x2, d2_0], dim=1))
        d3_1 = self.dec3_1(torch.cat([self.up(d2_1), x1, d3_0], dim=1))
        out2 = self.final2(d3_1)
        
        # Third decoder path (L=2)
        d3_2 = self.dec3_2(torch.cat([self.up(d2_1), x1, d3_0, d3_1], dim=1))
        out3 = self.final3(d3_2)
        
        # Final output (using deep supervision)
        if self.training:
            # During training, return all outputs for deep supervision
            return out1, out2, out3
        else:
            # During inference, return only the final output
            return out3

def test():
    # Test the model
    model = UNETpp(in_channels=3, out_channels=1)
    x = torch.randn((1, 3, 512, 512))
    if model.training:
        preds = model(x)
        for idx, pred in enumerate(preds):
            print(f"Output {idx+1} shape:", pred.shape)
    else:
        pred = model(x)
        print("Inference output shape:", pred.shape)

if __name__ == "__main__":
    test() 