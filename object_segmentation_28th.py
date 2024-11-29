import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
import cv2
import numpy as np
import torch

import cv2
import numpy as np
import torch

def user_select_object(predictions, image):
    """
    Allow user to interactively select an object from detected objects.
    
    Args:
    - predictions: Model predictions containing detected objects
    - image: Original image to display detection results
    
    Returns:
    - Index of the selected object box or None
    """
    # Convert image to numpy array if it's a PIL Image
    if hasattr(image, 'convert'):
        image = np.array(image.convert('RGB'))
    
    # Draw all detected objects with confidence > 0.8
    for i in range(len(predictions[0]['boxes'])):
        box = predictions[0]['boxes'][i].data.numpy()
        score = predictions[0]['scores'][i].data.numpy()
        
        if score > 0.8:
            # Draw rectangle for each detected object
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])), 
                (int(box[2]), int(box[3])), 
                (0, 255, 0),  # Green color
                2  # Line thickness
            )
    
    # Store the boxes for reference
    boxes = predictions[0]['boxes']
    
    # Global variables to track mouse selection
    clicked_point = None
    selected_index = None
    
    def on_mouse(event, x, y, flags, param):
        nonlocal clicked_point, selected_index
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point = (x, y)
            # Find which box contains the clicked point
            for i, box in enumerate(boxes):
                box_np = box.data.numpy()
                if (box_np[0] <= clicked_point[0] <= box_np[2] and 
                    box_np[1] <= clicked_point[1] <= box_np[3]):
                    selected_index = i
                    break
    
    # Show the image with detections
    cv2.namedWindow('Select Object', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Select Object', on_mouse)
    
    while True:
        cv2.imshow('Select Object', image)
        key = cv2.waitKey(1) & 0xFF
        
        # If an object is selected, close windows and return
        if selected_index is not None:
            cv2.destroyAllWindows()
            return selected_index
        
        # Exit if 'q' is pressed
        if key == ord('q'):
            cv2.destroyAllWindows()
            return None
    
    # This line is technically unreachable, but added for clarity
    return None

def get_model():
    # model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def detect_objects1(image_path, model):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)

    predictions = model(image_tensor)

    return predictions, image

def detect_objects(image_path, model):
    # Create a transform that includes normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
        #     std=[0.229, 0.224, 0.225]    # Standard ImageNet normalization
        # )
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

def draw_detected_objects(image, predictions):
    draw = ImageDraw.Draw(image)
    for element in range(len(predictions[0]['boxes'])):
        boxes = predictions[0]['boxes'][element].data.numpy()
        score = predictions[0]['scores'][element].data.numpy()
        if score > 0.8:  # Threshold can be adjusted
            draw.rectangle(((boxes[0], boxes[1]), (boxes[2], boxes[3])), outline ="red", width=3)

    return image

def draw_detected_objects1(image, predictions, confidence_threshold=0.8):
    # If image is a tensor, convert to PIL Image
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image.squeeze(0))
    
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    
    # Iterate through detected objects
    for i in range(len(predictions[0]['boxes'])):
        # Get box coordinates and confidence score
        boxes = predictions[0]['boxes'][i].detach().numpy()
        score = predictions[0]['scores'][i].detach().numpy()
        
        # Only draw boxes above the confidence threshold
        if score > confidence_threshold:
            # Draw rectangle
            draw.rectangle(
                [(boxes[0], boxes[1]), (boxes[2], boxes[3])], 
                outline="red", 
                width=3
            )
            
            # Optionally add confidence score as text
            label = f"{score:.2f}"
            draw.text((boxes[0], boxes[1]-10), label, fill="red")
    
    return image

def generate_mask_for_object(image_size, box):
    mask = Image.new('L', image_size, 255)  # Create a black image for mask
    draw = ImageDraw.Draw(mask)
    draw.rectangle(((box[0], box[1]), (box[2], box[3])), fill=0)
    return mask

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
    
    # Convert to PIL image
    mask = Image.fromarray((mask_data * 255).astype('uint8'), 'L')
    
    return mask

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

def fill_the_selected_mask(image, predictions, selected_mask_index, fill_color=(0, 255, 0, 255)):
    draw = ImageDraw.Draw(image, 'RGBA')  # Use 'RGBA' for transparency in fill

    for i, box in enumerate(predictions[0]['masks']):
        box = box.data.numpy()
        if i == selected_mask_index:  # Only fill the selected box
            # Draw the rectangle with or without fill
            fill = fill_color 
            draw.rectangle([box[0], box[1], box[2], box[3]], outline=None, fill=fill)
        
    return image

from PIL import Image, ImageDraw

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

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def display_masks(image, predictions):
    # Create a figure and axis for Matplotlib
    if predictions:  # Ensure there is at least one prediction
        prediction = predictions[0]  # Access the first (and only) image's predictions

        masks = prediction['masks']
        scores = prediction['scores']

        # Create a figure and axis for Matplotlib
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)

        # Only consider detections with a confidence > 0.5
        for mask, score in zip(masks, scores):
            if score > 0.5:
                # Convert the mask from a torch tensor to a numpy array
                mask_np = mask[0].mul(255).byte().cpu().numpy()
                # Overlay the mask on the image - simple binary mask
                ax.imshow(mask_np, alpha=0.5, cmap=ListedColormap(['none', 'red']))  # 'gray' can be replaced with any colormap

        plt.axis('off')  # Hide axes ticks
        plt.show()
    else:
        print("No predictions to display.")


def display_and_select_with_opencv(image, predictions):
    """
    Display an image with overlaid masks and allow the user to select an object by clicking on it.
    
    Args:
    image: A numpy array of the image (loaded and converted from PIL or similar).
    predictions: Output from a detection or segmentation model, including 'masks' and 'scores'.
    
    Returns:
    selected_index: The index of the selected object, or None if no selection is made.
    """
    # Assuming the image is in BGR format as expected by OpenCV
    # If the image is from PIL or another source, ensure it's in BGR format.
    # If it's in RGB format (e.g., from PIL), convert it: image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Create a copy of the image to overlay masks

    if predictions:  # Ensure there is at least one prediction
        prediction = predictions[0]  # Access the first (and only) image's predictions

        masks = prediction['masks']
        scores = prediction['scores']
        image_display = image.copy()

        for mask, score in zip(masks, scores):
            if score > 0.5:  # Adjust threshold as needed
                mask_np = mask.cpu().numpy()
                # Ensure mask is binary and matches the expected type and scale
                colored_mask = (mask_np > 0.5).astype(np.uint8) * 255  # Convert probabilities to binary mask
                mask_color = np.zeros_like(image, dtype=np.uint8)
                mask_color[:, :] = [0, 255, 0]  # Green color
                # Ensure mask is the correct size
                if colored_mask.shape[0:2] != image.shape[0:2]:
                    colored_mask = cv2.resize(colored_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                image_display = cv2.addWeighted(image_display, 1, cv2.bitwise_and(mask_color, mask_color, mask=colored_mask), 0.5, 0)

        cv2.namedWindow('Select Object', cv2.WINDOW_NORMAL)
        cv2.imshow('Select Object', image_display)

        selected_index = None

        def on_mouse(event, x, y, flags, param):
            nonlocal selected_index
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check each mask to see if the click is within the mask region
                for i, mask in enumerate(masks):
                    mask_np = mask.cpu().numpy()
                    if mask_np[y, x] > 0.5:
                        selected_index = i
                        print(f"Selected object index: {selected_index}")
                        cv2.destroyAllWindows()
                        return  # Exit after selection

        cv2.setMouseCallback('Select Object', on_mouse)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if selected_index is not None or key == ord('q'):
                break

        cv2.destroyAllWindows()
        return selected_index

def display_and_select_with_opencv(image_np, predictions):
    # Assuming `image_np` is the correctly converted numpy array from the PIL Image
    image_display = image_np.copy()
    prediction = predictions[0]
    masks = prediction['masks']
    scores = prediction['scores']
    
    for mask, score in zip(masks, scores):
        if score > 0.5:  # Adjust the threshold as needed
            #mask_np = mask.cpu().numpy()
            # Ensure mask is binary and matches the expected type and scale
            #colored_mask = (mask_np > 0.5).astype(np.uint8) * 255  # Convert to binary mask, scaled to 0-255
            mask_np = mask.cpu().numpy().squeeze()  # Remove extra dimensions
            colored_mask = (mask_np > 0.5).astype(np.uint8) * 255
            
            # Ensure the mask is the correct size
            if colored_mask.ndim == 2:  # Ensure it's a 2D mask
                if colored_mask.shape[0] != image_np.shape[0] or colored_mask.shape[1] != image_np.shape[1]:
                    colored_mask = cv2.resize(colored_mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Create a colored overlay (e.g., green) for the mask
            mask_color = np.zeros_like(image_np, dtype=np.uint8)
            mask_color[:, :] = [0, 255, 0]  # Green color
            image_display = cv2.addWeighted(image_display, 1, cv2.bitwise_and(mask_color, mask_color, mask=colored_mask), 0.5, 0)

    # Display the image with detections
    cv2.namedWindow('Select Object', cv2.WINDOW_NORMAL)
    cv2.imshow('Select Object', image_display)
    cv2.imwrite('detected_image.jpg', image_display)  # Specify the file path and name`
    selected_index = None

    def on_mouse(event, x, y, flags, param):
        nonlocal selected_index
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, mask in enumerate(masks):
                # Convert mask to numpy and ensure it matches image dimensions
                mask_np = mask.cpu().numpy().squeeze()
                
                # Resize mask to match image dimensions if necessary
                if mask_np.shape != image_np.shape[:2]:
                    mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                
                # Convert to binary mask
                mask_binary = (mask_np > 0.5).astype(np.uint8)
                
                # Check if clicked point is within mask
                if mask_binary[y, x] == 1:
                    selected_index = i
                    cv2.destroyAllWindows()
                    return

    cv2.setMouseCallback('Select Object', on_mouse)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if selected_index is not None or key == ord('q'):
            break

    cv2.destroyAllWindows()
    return selected_index

def object_detection_and_selection():
    detection_model = get_model()
    image_path = 'data/me1.png'
    predictions, original_image = detect_objects(image_path, detection_model)
    image = Image.open(image_path).convert("RGB")

    image_np = np.array(image.convert('RGB'))
    #display_masks(image, predictions)
    selected_index = display_and_select_with_opencv(image_np, predictions)
    # detected_image = draw_detected_objects(original_image.copy(), predictions)
    
    # Use the new interactive selection method
    #selected_index = user_select_object(predictions, original_image)
    print("selected index is:", selected_index)

    if selected_index is not None:
        # Get the selected box using the index
        selected_mask = predictions[0]['masks'][selected_index].data.numpy()
        edited_image = fill_the_selected_mask(original_image, predictions, selected_index)
        edited_image_rgb = edited_image.convert('RGB')  # Convert RGBA to RGB
        edited_image_rgb.save('edited_image.jpg')
        # Save or display the detected objects image
        # detected_image.save('detected_image.jpg')
        mask = generate_mask_for_object(original_image.size, selected_mask)
        mask.save('selected_mask.png')
        return 'edited_image.jpg', 'selected_mask.png'

# def user_select_object(predictions):
#     # Assuming there's some interface for users to select, this function
#     # would return the bounding box of the selected object
#     # For demonstration, automatically choose the first detected object
#     return predictions[0]['boxes'][0].data.numpy()

if __name__ == "__main__":
    # detection_model = get_model()
    # image_path = 'data/me.png'
    # predictions, original_image = detect_objects(image_path, detection_model)
    # detected_image = draw_detected_objects(original_image.copy(), predictions)

    # # Let user select an object to remove (this part needs interactive selection)
    # selected_box = user_select_object(predictions)
    # mask = generate_mask_for_object(original_image.size, selected_box)

    # # Save or display the detected objects image
    # detected_image.save('detected_image.jpg')
    # mask.save('selected_mask.png')

    # Proceed with inpainting using the generated mask
    # test_inpainting(image_path, 'selected_mask.png')

    object_detection_and_selection()
