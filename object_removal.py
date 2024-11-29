import cv2
import torch
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

class ObjectRemover:
    def __init__(self):
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def detect_boundaries(self, image):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)[0]

        high_conf_idx = predictions['scores'] > 0.5
        masks = predictions['masks'][high_conf_idx]
        boxes = predictions['boxes'][high_conf_idx]

        return masks, boxes

    def remove_object(self, image, mask, inpaint_radius=3):
        try:
            # Convert mask to binary format
            binary_mask = (mask > 0.5).cpu().numpy().astype(np.uint8)[0, 0]
            binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))
            
            # Display the binary mask for debugging
            cv2.imshow("Binary Mask", binary_mask * 255)
            cv2.waitKey(0)

            # Dilate and invert the mask if needed
            kernel = np.ones((5, 5), np.uint8)
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

            # Optionally invert the mask
            if np.mean(binary_mask) > 0.5:
                binary_mask = 1 - binary_mask

            # Perform inpainting
            result_image = cv2.inpaint(image, binary_mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)

            # Display inpainting result
            cv2.imshow("Inpainting Result", result_image)
            cv2.waitKey(0)

            return result_image
        except Exception as e:
            print(f"Error in remove_object: {str(e)}")
            return image

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return

        height, width = image.shape[:2]
        
        # Detect object boundaries
        masks, boxes = self.detect_boundaries(image)
        
        if len(masks) == 0:
            print("No objects detected!")
            return

        window_name = "Object Removal Tool"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

        # Store the original and working images
        original_image = image.copy()
        working_image = image.copy()
        current_image = image.copy()
        selected_objects = set()

        def draw_boundaries(img, selected):
            result = img.copy()
            for idx, box in enumerate(boxes):
                box = box.cpu().numpy().astype(int)
                color = (0, 255, 0) if idx in selected else (255, 0, 0)
                cv2.rectangle(result, (box[0], box[1]), (box[2], box[3]), color, 2)
            return result

        def mouse_callback(event, x, y, flags, param):
            nonlocal current_image, selected_objects
            
            if event == cv2.EVENT_LBUTTONDOWN:
                for idx, box in enumerate(boxes):
                    box = box.cpu().numpy().astype(int)
                    if (box[0] <= x <= box[2] and box[1] <= y <= box[3]):
                        if idx in selected_objects:
                            selected_objects.remove(idx)
                        else:
                            selected_objects.add(idx)
                        
                        # Show boundaries on working image
                        current_image = draw_boundaries(working_image.copy(), selected_objects)
                        cv2.imshow(window_name, current_image)

        cv2.setMouseCallback(window_name, mouse_callback)

        print("\nInstructions:")
        print("- Click on objects to select/deselect them")
        print("- Press 'r' to remove selected objects")
        print("- Press 'u' to undo and restore original image")
        print("- Press 's' to save the result")
        print("- Press 'q' to quit")

        # Show initial image with boundaries
        current_image = draw_boundaries(image.copy(), selected_objects)
        
        while True:
            cv2.imshow(window_name, current_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                if selected_objects:
                    # Remove selected objects
                    working_image = image.copy()
                    for idx in selected_objects:
                        working_image = self.remove_object(working_image, masks[idx])
                    
                    # Update the display
                    current_image = draw_boundaries(working_image.copy(), selected_objects)
                    cv2.imshow(window_name, current_image)
                    
                    # Clear selections after removal
                    selected_objects.clear()
            elif key == ord('u'):
                # Restore original image
                working_image = image.copy()
                selected_objects.clear()
                current_image = draw_boundaries(working_image.copy(), selected_objects)
            elif key == ord('s'):
                # Save the result
                output_path = image_path.rsplit('.', 1)[0] + '_removed.jpg'
                cv2.imwrite(output_path, working_image)
                print(f"Result saved as: {output_path}")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    remover = ObjectRemover()
    remover.process_image("./data/kitchen.png")  # Replace with your image path
