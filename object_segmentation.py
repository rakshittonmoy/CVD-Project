import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
import cv2
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def get_model():
    model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def detect_objects(image_path, model):
    # Create a transform that includes normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Open the image
    image = Image.open(image_path)
    
    # If the image is RGBA (4 channels), convert to RGB
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Apply the transform and add batch dimension
    image_tensor = transform(image).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)

    return predictions, image


def generate_mask_for_object(image_size, mask_data):
    # Ensure mask_data is a numpy array and has the correct dimensions
    mask_data = np.array(mask_data).squeeze()
    
    # Ensure mask is 2D
    if mask_data.ndim > 2:
        mask_data = mask_data.squeeze(0)
    
    # Convert to binary mask if not already
    mask_data = (mask_data > 0.5).astype(np.uint8)
    
    # Resize mask to match image size
    if mask_data.shape[0] != image_size[1] or mask_data.shape[1] != image_size[0]:
        mask_data = cv2.resize(mask_data, (image_size[0], image_size[1]), interpolation=cv2.INTER_NEAREST)
    
    # Invert the mask so selected area is black (0)
    mask_data = 1 - mask_data
    
    # Convert to PIL image
    mask = Image.fromarray((mask_data * 255).astype('uint8'), 'L')
    
    return mask


def fill_the_selected_mask(image, predictions, selected_mask_index, fill_color=(0, 255, 0, 128)):
    """
    Fill the selected mask in the image using RGBA for transparency.
    
    Args:
    - image: PIL Image object in 'RGB' mode.
    - predictions: Output from a segmentation model, should include masks.
    - selected_mask_index: Index of the mask to be filled.
    - fill_color: Tuple representing the RGBA color value to fill.
    
    Returns:
    - Image object with the selected mask filled.
    """
    # Ensure image is in RGBA mode for transparency handling
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # Extract the specific mask based on the selected index
    selected_mask = predictions[0]['masks'][selected_mask_index].cpu().numpy()
    thresholded_mask = (selected_mask > 0.5).astype(np.uint8)  # Ensure binary mask

    # Check if the mask is 3D and convert it to 2D by removing the channel dimension if necessary
    if thresholded_mask.ndim == 3 and thresholded_mask.shape[0] == 1:
        thresholded_mask = thresholded_mask.squeeze(0)  # Remove the singleton dimension

    # Convert the binary mask to a PIL image
    mask_image = Image.fromarray(thresholded_mask * 255, mode='L')
    
    # Create a new image with the same size as the original filled with the fill color
    filled_image = Image.new('RGBA', image.size, fill_color)
    
    # Composite the filled image with the original using the mask
    image_with_filled_mask = Image.composite(filled_image, image, mask_image)

    return image_with_filled_mask


def display_and_select_with_matplotlib(image_np, predictions):
    # Assuming `image_np` is the correctly converted numpy array from the PIL Image
    image_display = image_np.copy()
    prediction = predictions[0]
    masks = prediction['masks']
    scores = prediction['scores']
    
    # Initialize selected index
    selected_index = None

    # Create the overlayed image
    for mask, score in zip(masks, scores):
        if score > 0.5:  # Adjust the threshold as needed
            mask_np = mask.cpu().numpy().squeeze()  # Remove extra dimensions
            colored_mask = (mask_np > 0.5).astype(np.uint8) * 255
            
            # Resize the mask to match the image dimensions
            if colored_mask.shape != image_np.shape[:2]:
                colored_mask = cv2.resize(colored_mask, (image_np.shape[1], image_np.shape[0]), 
                                          interpolation=cv2.INTER_NEAREST)
            
            # Create a colored overlay (e.g., green) for the mask
            mask_color = np.zeros_like(image_np, dtype=np.uint8)
            mask_color[:, :] = [0, 255, 0]  # Green color
            image_display = cv2.addWeighted(image_display, 1, 
                                            cv2.bitwise_and(mask_color, mask_color, mask=colored_mask), 
                                            0.5, 0)

    # Plot the image with Matplotlib
    fig, ax = plt.subplots()
    ax.imshow(image_display)
    ax.set_title("Click on an object to select (Close to quit)")
    plt.axis('off')
    plt.savefig("detected_image.jpg")
    plt.axis('off')

    def on_mouse(event):
        nonlocal selected_index
        if event.xdata is None or event.ydata is None:
            return  # Ignore clicks outside the image
        x, y = int(event.xdata), int(event.ydata)

        for i, mask in enumerate(masks):
            mask_np = mask.cpu().numpy().squeeze()
            if mask_np.shape != image_np.shape[:2]:
                mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask_np > 0.5).astype(np.uint8)
            if mask_binary[y, x] == 1:  # Check if the clicked point is in the mask
                selected_index = i
                print(f"Object {i} selected at ({x}, {y}).")
                plt.close()  # Close the plot
                return

    # Connect the mouse event
    cid = fig.canvas.mpl_connect('button_press_event', on_mouse)

    # Show the plot and wait for interaction
    plt.show()

    return selected_index

def object_detection_and_selection():
    detection_model = get_model()
    image_path = 'data/chikkamaglur.png'
    predictions, original_image = detect_objects(image_path, detection_model)
    image = Image.open(image_path).convert("RGB")

    image_np = np.array(image.convert('RGB'))
    selected_index = display_and_select_with_matplotlib(image_np, predictions)
    
    # Use the new interactive selection method
    print("selected index is:", selected_index)

    if selected_index is not None:
        # Get the selected box using the index
        selected_mask = predictions[0]['masks'][selected_index].data.numpy()
        edited_image = fill_the_selected_mask(original_image, predictions, selected_index)
        edited_image_rgb = edited_image.convert('RGB')  # Convert RGBA to RGB
        edited_image_rgb.save('edited_image.jpg')
        mask = generate_mask_for_object(original_image.size, selected_mask)
        mask.save('selected_mask.png')
        return 'edited_image.jpg', 'selected_mask.png'

if __name__ == "__main__":
    object_detection_and_selection()
