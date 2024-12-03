from object_segmentation_28th import object_detection_and_selection
import torch
import numpy as np
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from io import BytesIO
import cv2
import numpy as np
import queue
import os
import datetime
import torchvision.transforms.functional as F



drawing = False  # True if the mouse is pressed. False otherwise
ix, iy = -1, -1 
results_queue = queue.Queue()


def get_image_dimensions(image_path):
    """
    Get the original dimensions of an image
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    
    Returns:
    --------
    tuple
        (width, height) of the original image
    """
    with Image.open(image_path) as img:
        return img.size  # Returns (width, height)

def smart_resize(image, max_size=1024):
    """
    Resize image intelligently while preserving aspect ratio
    
    Parameters:
    -----------
    image : torch.Tensor
        Input image tensor
    max_size : int, optional
        Maximum dimension size (default 1024)
    
    Returns:
    --------
    torch.Tensor
        Resized image tensor
    """
    # Convert tensor to PIL Image for easier manipulation
    pil_image = torchvision.transforms.ToPILImage()(image.squeeze(0))
    
    # Calculate resize dimensions while maintaining aspect ratio
    width, height = pil_image.size
    
    # Determine scaling factor
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Use high-quality resampling
        resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    else:
        resized_image = pil_image
    
    # Convert back to tensor
    return torchvision.transforms.ToTensor()(resized_image).unsqueeze(0)

def preprocess_image(image_path, mask_path=None):
    """
    Preprocess image and mask with smart resizing
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    mask_path : str, optional
        Path to the mask image
    
    Returns:
    --------
    tuple
        (processed image tensor, processed mask tensor)
    """
    # Get original image dimensions
    original_dims = get_image_dimensions(image_path)
    print(f"Original image dimensions: {original_dims}")
    
    # Open and convert image
    image = Image.open(image_path).convert('RGB')
    image_tensor = torchvision.transforms.ToTensor()(image)
    
    # Smart resize
    resized_image = smart_resize(image_tensor)
    
    # Process mask similarly if provided
    if mask_path:
        mask = Image.open(mask_path).convert('L')
        mask_tensor = torchvision.transforms.ToTensor()(mask)
        resized_mask = smart_resize(mask_tensor)
    else:
        # Create a default mask if no mask provided
        resized_mask = torch.ones_like(resized_image[0:1])
    
    return resized_image, resized_mask

def inpaint_image(image, mask, queue_results=None, num_iterations=800, learning_rate=0.01):
    """
    Perform inpainting with more flexible resizing
    
    Parameters:
    -----------
    image : torch.Tensor
        Input image tensor
    mask : torch.Tensor
        Mask tensor
    queue_results : Queue, optional
        For tracking intermediate results
    num_iterations : int, optional
        Number of training iterations
    learning_rate : float, optional
        Optimization learning rate
    
    Returns:
    --------
    torch.Tensor
        Inpainted image tensor
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and optimizer
    net = DeepImagePrior().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    # Create input noise matching image dimensions
    input_noise = torch.randn(1, 32, image.shape[2], image.shape[3]).to(device)
    input_noise.requires_grad_(True)
    
    # Training loop with optional result tracking
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # Generate output
        out = net(input_noise)
        
        # Compute loss only on masked regions
        loss = torch.mean((out - image) ** 2 * mask)
        loss.backward()
        
        optimizer.step()
        
        # Optional result tracking
        if queue_results and i % 100 == 0:
            print(f'Iteration {i}/{num_iterations}, Loss: {loss.item():.6f}')
            queue_results.put((out.clone().detach().cpu(), f'Iteration {i}: Loss={loss.item():.6f}'))

    return out.detach()

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


def test_inpainting(image_path, mask_path):
    # Load and preprocess image and mask
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    
    # Ensure image and mask have the same size
    image, mask = ensure_same_size(image, mask)
    
    image = transforms.ToTensor()(image)
    mask = transforms.ToTensor()(mask)

    image, mask = preprocess_image(image_path, mask_path)
    
    # Run inpainting
    print("Starting inpainting process...")
    result, losses = inpaint_image(image, mask, results_queue, num_iterations=5000, learning_rate=0.1)
    
    initial_image = load_initial_image('detected_image.jpg')
    display_results(initial_image)
    display_loss_graph(losses)
    # Save result as PNG for best quality
    save_result(result, "inpainting_result.png")
    print("Inpainting complete! Result saved as 'inpainting_result.png'")



def load_initial_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def display_loss_graph(losses):
    # Plotting the loss graph with annotations
    plt.figure(figsize=(20, 20))
    plt.plot(losses, label='Loss per Iteration')
    plt.title('Loss during Training')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    # Annotating specific points
    for i in range(0, len(losses), 200):  # Every 50 iterations
        plt.annotate(f'{losses[i]:.4f}',  # The loss value formatted to 4 decimals
                    (i, losses[i]),      # Point to annotate
                    textcoords="offset points",  # Positioning of the text
                    xytext=(0,20),       # Distance from the point
                    ha='center')         # Alignment

    plt.xticks(range(0, len(losses), 200))  # Set x-ticks to show every 50 iterations
    plt.legend()
    results_dir = "results"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{results_dir}/loss_plot_{timestamp}.png"
    plt.savefig(filename)
    plt.show()

def display_results(initial_image):
    # Convert the initial image for plotting
    initial_img = initial_image.squeeze(0).permute(1, 2, 0).numpy()

    # Determine the total number of results to configure the grid of subplots
    num_results = results_queue.qsize() + 1  # +1 for the initial image
    cols = int(np.ceil(np.sqrt(num_results)))
    rows = int(np.ceil(num_results / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    fig.tight_layout(pad=3.0)

    # Flatten axes array if necessary
    if num_results > 1:
        axes = axes.flatten()

    # Display the initial image first
    axes[0].imshow(initial_img)
    axes[0].set_title("Initial Image")
    axes[0].axis('off')

    # Iterate through results in the queue and plot them
    for i in range(1, num_results):  # Start from 1 to leave space for the initial image
        if not results_queue.empty():
            result, info = results_queue.get()
            img = result.squeeze(0).permute(1, 2, 0)  # Adjust dimensions for plotting
            axes[i].imshow(img)
            axes[i].set_title(info)
            axes[i].axis('off')

    # If we have more subplots than images, turn off the extra axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')



    # Create the results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Generate a timestamp and save the figure
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{results_dir}/final_plot_{timestamp}.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

    plt.show()

def ensure_same_size(image, mask):
    """Ensure the image and mask have the same size"""
    if image.size != mask.size:
        mask = mask.resize(image.size)
    return image, mask


# Now, `mask` contains the user-generated mask, and `img` shows where the user has drawn.
    
if __name__ == "__main__":
    # Import the DeepImagePrior class and related functions from previous code
    from deep_image_prior import DeepImagePrior, inpaint_image, save_result
    import os
    
    # Load an image
    img = cv2.imread('data/cow.png')

    import torch

    # Check if CUDA is available
    print("CUDA available:", torch.cuda.is_available())

    # Check the active device
    print("Current device:", torch.cuda.current_device())

    # Check the device name
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    image_path, selected_mask = object_detection_and_selection()
    
    # Run the test
    test_inpainting(image_path, selected_mask)