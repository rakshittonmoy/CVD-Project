import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms
import requests
from io import BytesIO
import cv2
import numpy as np
import threading

drawing = False  # True if the mouse is pressed. False otherwise
ix, iy = -1, -1 

def run_inpainting(image_path, mask_path):
    print("Starting inpainting process...")

    test_inpainting(image_path, mask_path)
    # Placeholder for inpainting call
    # simulate a long-running task
    #import time; time.sleep(5)
    print("Inpainting complete!")

def create_test_mask(image_size, object_position, object_size):
    """
    Create a simple rectangular mask for testing
    Returns a binary mask as a PIL Image in 'L' (grayscale) mode
    """
    mask = Image.new('L', image_size, 255)  # White background (keep)
    draw = ImageDraw.Draw(mask)
    
    # Draw black rectangle (area to inpaint)
    x1 = object_position[0] - object_size[0]//2
    y1 = object_position[1] - object_size[1]//2
    x2 = x1 + object_size[0]
    y2 = y1 + object_size[1]
    draw.rectangle([x1, y1, x2, y2], fill=0)  # Black rectangle (remove)
    
    # Ensure mask is binary (0 or 255)
    mask = np.array(mask)
    mask = ((mask > 127) * 255).astype(np.uint8)
    mask = Image.fromarray(mask)
    
    return mask

def apply_mask_to_image(image, mask):
    """
    Apply mask to image to create the masked version
    White in mask (255) = keep original
    Black in mask (0) = mask out (replace with gray)
    """
    # Convert images to numpy arrays
    img_arr = np.array(image).astype(np.float32) / 255.0
    mask_arr = np.array(mask).astype(np.float32) / 255.0
    
    # Expand mask to 3 channels if needed
    if len(mask_arr.shape) == 2:
        mask_arr = np.expand_dims(mask_arr, -1)
        mask_arr = np.repeat(mask_arr, 3, axis=-1)
    
    # Create gray color for masked regions
    gray_color = np.array([0.5, 0.5, 0.5])  # Mid-gray
    
    # Apply mask: keep original image where mask is 1, use gray where mask is 0
    masked_image = img_arr * mask_arr + gray_color * (1 - mask_arr)
    
    return (masked_image * 255).astype(np.uint8)

def download_test_image(url):
    """Download a test image from URL and convert to RGB mode"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

def inpaint_image(original_image_path, masked_image_path, num_iterations=3000, learning_rate=0.01, save_progress=True):
    """
    Perform inpainting using Deep Image Prior
    Args:
        original_image_path: Path to the original image
        masked_image_path: Path to the image with masked regions
        save_progress: If True, save intermediate results during training
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load images
    original_image = Image.open(original_image_path).convert('RGB')
    masked_image = Image.open(masked_image_path).convert('RGB')
    
    # Convert to tensors
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    original_tensor = transform(original_image).to(device)
    masked_tensor = transform(masked_image).to(device)
    
    # Create binary mask
    binary_mask = create_binary_mask(
        masked_tensor.permute(1, 2, 0).cpu().numpy(),
        original_tensor.permute(1, 2, 0).cpu().numpy()
    )
    mask_tensor = torch.FloatTensor(binary_mask).unsqueeze(0).to(device)
    
    # Initialize model and optimizer
    net = DeepImagePrior().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    # Create input noise
    input_noise = torch.randn(1, net.noise_channels, 256, 256).to(device)
    
    # Lists to store progress
    losses = []
    intermediate_results = []
    
    # Training loop
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        out = net(input_noise)
        
        # Compute loss only on non-masked regions
        loss = torch.mean((out - original_tensor) ** 2 * mask_tensor)
        loss.backward()
        
        optimizer.step()
        
        losses.append(loss.item())
        
        # Save intermediate results
        if save_progress and (i + 1) % 100 == 0:
            intermediate_results.append(out.detach().cpu().squeeze(0).permute(1, 2, 0).numpy())
            print(f'Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6f}')
    
    # Visualize training progress
    if save_progress:
        visualize_training_progress(
            original_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy(),
            masked_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy(),
            binary_mask,
            intermediate_results,
            losses
        )
    
    return out.detach().cpu()

def visualize_process(original, mask, masked, result):
    """Visualize the original, mask, masked image, and result"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Mask (Black = Inpaint)')
    axes[1].axis('off')
    
    axes[2].imshow(masked)
    axes[2].set_title('Masked Image')
    axes[2].axis('off')
    
    axes[3].imshow(result)
    axes[3].set_title('Inpainting Result')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_training_progress(original, masked, mask, intermediate_results, losses):
    """Visualize the training progress including loss curve and intermediate results"""
    num_results = len(intermediate_results)
    fig = plt.figure(figsize=(20, 10))
    
    # Plot original, masked, and mask
    plt.subplot(2, 4, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(masked)
    plt.title('Masked Image')
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('Binary Mask\n(White=Keep, Black=Inpaint)')
    plt.axis('off')
    
    # Plot final result
    plt.subplot(2, 4, 4)
    plt.imshow(intermediate_results[-1])
    plt.title('Final Result')
    plt.axis('off')
    
    # Plot loss curve
    plt.subplot(2, 1, 2)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()
    
    # Create gif of intermediate results
    create_progress_gif(intermediate_results)

def create_progress_gif(intermediate_results):
    """Create a gif showing how the inpainting progresses"""
    frames = []
    for img in intermediate_results:
        # Convert to PIL Image
        img = Image.fromarray((img * 255).astype(np.uint8))
        frames.append(img)
    
    # Save as gif
    frames[0].save(
        'inpainting_progress.gif',
        save_all=True,
        append_images=frames[1:],
        duration=200,  # milliseconds per frame
        loop=0
    )

def test_inpainting(image_path, mask_path):
    # Load and preprocess image and mask
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    
    # Ensure image and mask have the same size
    image, mask = ensure_same_size(image, mask)
    
    image = transforms.ToTensor()(image)
    mask = transforms.ToTensor()(mask)
    
    # Run inpainting
    print("Starting inpainting process...")
    result = inpaint_image(image, mask, num_iterations=2000, learning_rate=0.01)
    
    # Visualize results
    visualize_process(
        image.squeeze(0).permute(1, 2, 0).numpy(),
        mask.squeeze(0).numpy(),
        apply_mask_to_image(image.squeeze(0).permute(1, 2, 0).numpy(), mask.squeeze(0).numpy()),
        result.squeeze(0).permute(1, 2, 0).numpy()
    )
    
    # Save result as PNG for best quality
    save_result(result, "inpainting_result.png")
    print("Inpainting complete! Result saved as 'inpainting_result.png'")

def ensure_same_size(image, mask):
    """Ensure the image and mask have the same size"""
    if image.size != mask.size:
        mask = mask.resize(image.size)
    return image, mask

# Function to handle mouse events
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, mask

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            cv2.rectangle(mask, (ix, iy), (x, y), 0, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        cv2.rectangle(mask, (ix, iy), (x, y), 0, -1)


# Now, `mask` contains the user-generated mask, and `img` shows where the user has drawn.
    
if __name__ == "__main__":
    # Import the DeepImagePrior class and related functions from previous code
    from deep_image_prior import DeepImagePrior, inpaint_image, save_result
    import os
    
    # Load an image
    img = cv2.imread('data/test.png')
    mask = np.ones_like(img) * 255  # Create a mask initialized to white

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)

    while(1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key to stop
            break

    cv2.destroyAllWindows()

    # Save the edited image and the mask
    cv2.imwrite('edited_image.jpg', img)  # Save the image with user-drawn rectangles
    cv2.imwrite('mask.png', mask)  # Save the binary mask

    print("Images saved: 'edited_image.jpg' and 'mask.png'")

    # Run the inpainting in a separate thread
    threading.Thread(target=run_inpainting, args=('edited_image.jpg', 'mask.png'), daemon=True).start()