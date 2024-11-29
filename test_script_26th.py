from object_segmentation_28th import object_detection_and_selection
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms
import requests
from io import BytesIO
import cv2
import numpy as np
import queue
import os
import datetime

drawing = False  # True if the mouse is pressed. False otherwise
ix, iy = -1, -1 
results_queue = queue.Queue()

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
    
    # Run inpainting
    print("Starting inpainting process...")
    result = inpaint_image(image, mask, results_queue, num_iterations=3000, learning_rate=0.01)
    
    # Visualize results
    # visualize_process(
    #     image.squeeze(0).permute(1, 2, 0).numpy(),
    #     mask.squeeze(0).numpy(),
    #     apply_mask_to_image(image.squeeze(0).permute(1, 2, 0).numpy(), mask.squeeze(0).numpy()),
    #     result.squeeze(0).permute(1, 2, 0).numpy()
    # )
    initial_image = load_initial_image('detected_image.jpg')
    display_results(initial_image)
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
    from deep_image_prior_28th import DeepImagePrior, inpaint_image, save_result
    import os
    
    # Load an image
    img = cv2.imread('data/cow.png')
    # mask = np.ones_like(img) * 255  # Create a mask initialized to white

    # cv2.namedWindow('image')
    # cv2.setMouseCallback('image', draw_rectangle)

    # while(1):
    #     cv2.imshow('image', img)
    #     k = cv2.waitKey(1) & 0xFF
    #     if k == 27:  # ESC key to stop
    #         break

    # cv2.destroyAllWindows()

    # Save the edited image and the mask
    # cv2.imwrite('edited_image.jpg', img)  # Save the image with user-drawn rectangles
    # cv2.imwrite('mask.png', mask)  # Save the binary mask

    # print("Images saved: 'edited_image.jpg' and 'mask.png'")

    image_path, selected_mask = object_detection_and_selection()
    
    # Run the test
    test_inpainting(image_path, selected_mask)