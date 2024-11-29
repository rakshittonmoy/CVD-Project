import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # return self.double_conv(x)
        residual = x
        out = self.double_conv(x)
        out += residual
        return out

class DeepImagePrior(nn.Module):
    def __init__(self, input_channels=32, output_channels=3):
        super().__init__()
        self.noise_channels = input_channels
        
        # Encoder
        self.enc1 = DoubleConv(input_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Decoder
        self.dec4 = DoubleConv(512, 256)
        self.dec3 = DoubleConv(512, 128)
        self.dec2 = DoubleConv(256, 64)
        self.dec1 = DoubleConv(128, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, output_channels, kernel_size=1)
        
        # Max pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool(enc3)
        
        # Bottleneck
        x = self.enc4(x)
        x = self.dec4(x)
        
        # Decoder
        x = self.upsample(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upsample(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upsample(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        return torch.sigmoid(self.final(x))

def preprocess_image(image_path, mask_path=None):
    """Preprocess image and mask for inpainting"""
    image = Image.open(image_path).convert('RGB')
    image = torchvision.transforms.ToTensor()(image)
    
    if mask_path:
        mask = Image.open(mask_path).convert('L')
        mask = torchvision.transforms.ToTensor()(mask)
    else:
        mask = torch.ones_like(image[0:1])
    
    return image, mask

def inpaint_image(image, mask, queue_results, num_iterations=3000, learning_rate=0.01,):
    """Perform inpainting using Deep Image Prior"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Resize image and mask to match model input size
    model_input_size = (256, 256)
    image = F.interpolate(image.unsqueeze(0), size=model_input_size, mode='bilinear', align_corners=True)
    mask = F.interpolate(mask.unsqueeze(0), size=model_input_size, mode='bilinear', align_corners=True)
    
    # Create model and optimizer
    net = DeepImagePrior().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    # Create input noise
    input_noise = torch.randn(1, net.noise_channels, image.shape[2], image.shape[3]).to(device)
    
    # Lists to store progress
    losses = []
    intermediate_results = []

    # Training loop
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        out = net(input_noise)
        
        # Compute loss only on non-masked regions
        loss = torch.mean((out - image) ** 2 * mask)
        loss.backward()
        
        optimizer.step()

        losses.append(loss.item())
        
        if i % 100 == 0:
            print(f'Iteration {i}/{num_iterations}, Loss: {loss.item():.6f}')
            queue_results.put((out.clone().detach().cpu(), f'Iteration {i}: Loss={loss.item():.6f}'))

    return out.detach()

def save_result(tensor, save_path):
    """Save the result as an image"""
    result = tensor.squeeze(0).permute(1, 2, 0).numpy()
    result = np.clip(result, 0, 1)
    plt.imsave(save_path, result)

# Example usage
if __name__ == "__main__":
    image_path = "input_image.jpg"
    mask_path = "mask.png"
    output_path = "inpainted_result.png"
    
    result = inpaint_image(image_path, mask_path)
    save_result(result, output_path)