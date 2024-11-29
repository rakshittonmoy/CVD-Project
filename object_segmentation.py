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

def fill_the_selected_box(image, predictions, selected_box_index, fill_color=(0, 255, 0, 255)):
    draw = ImageDraw.Draw(image, 'RGBA')  # Use 'RGBA' for transparency in fill

    for i, box in enumerate(predictions[0]['boxes']):
        box = box.data.numpy()
        if i == selected_box_index:  # Only fill the selected box
            # Draw the rectangle with or without fill
            fill = fill_color 
            draw.rectangle([box[0], box[1], box[2], box[3]], outline=None, fill=fill)
        
    return image

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


def object_detection_and_selection():
    detection_model = get_model()
    image_path = 'data/test_image1.jpg'
    predictions, original_image = detect_objects(image_path, detection_model)
    image = Image.open(image_path).convert("RGB")
    display_masks(image, predictions)
    # detected_image = draw_detected_objects(original_image.copy(), predictions)
    
    # # Use the new interactive selection method
    selected_index = user_select_object(predictions, original_image)
    print("selected index is:", selected_index)

    if selected_index is not None:
        # Get the selected box using the index
        selected_box = predictions[0]['boxes'][selected_index].data.numpy()
        edited_image = fill_the_selected_box(original_image, predictions, selected_index)
        edited_image.save('edited_image.jpg')
    #     # Save or display the detected objects image
    #     detected_image.save('detected_image.jpg')
    #     mask = generate_mask_for_object(original_image.size, selected_box)
    #     mask.save('selected_mask.png')
    #     return 'edited_image.jpg', 'selected_mask.png'

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
