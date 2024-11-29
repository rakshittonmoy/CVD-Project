import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DeepImagePrior(nn.Module):
    def __init__(self, input_channels=32, output_channels=3):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 1024)

        # Decoder with skip connections
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ConvBlock(512 + 256, 256)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(256 + 128, 128)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock(128 + 64, 64)

        # Final layer
        self.final = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        x = self.enc4(x)
        
        # Decoder with skip connections
        x = self.upsample3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upsample2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upsample1(x)
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

# Combined loss function
def custom_loss(out, image, mask):
    # Masked reconstruction loss
    reconstruction_loss = torch.mean((out - image) ** 2 * mask)
    
    # Total variation loss to smooth the result
    tv_loss = torch.mean(torch.abs(out[:, :, :, :-1] - out[:, :, :, 1:])) + \
                torch.mean(torch.abs(out[:, :, :-1, :] - out[:, :, 1:, :]))
    
    return reconstruction_loss + 0.1 * tv_loss

def inpaint_image(image, mask, queue_results=None, num_iterations=3000, learning_rate=0.01):
    """Perform inpainting using Deep Image Prior"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Resize image and mask to match model input size
    model_input_size = (256, 256)
    image = F.interpolate(image.unsqueeze(0), size=model_input_size, mode='bicubic', align_corners=True)
    mask = F.interpolate(mask.unsqueeze(0), size=model_input_size, mode='nearest')
    
    # Create model and optimizer
    net = DeepImagePrior().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    # Create input noise with correct number of channels
    input_noise = torch.randn(1, 32, image.shape[2], image.shape[3]).to(device)
    input_noise.requires_grad_(True)

    
    # Training loop
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # Generate output
        out = net(input_noise)
        
        # Compute loss only on masked regions
        # loss = torch.mean((out - image) ** 2 * mask)
        loss = custom_loss(out, image, mask)
        loss.backward()
        
        optimizer.step()
        
        if queue_results and i % 100 == 0:
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
    
    image, mask = preprocess_image(image_path, mask_path)
    result = inpaint_image(image, mask)
    save_result(result, output_path)