# Object Detection, Erasure, and Seamless Inpainting Framework

This project automates the detection and removal of objects from images, facilitating minimal user intervention. Utilizing **Mask R-CNN** for precise object segmentation and **Deep Image Prior** for advanced inpainting, the framework allows users to seamlessly remove unwanted elements and restore the visual continuity of images. This solution combines state-of-the-art techniques in object detection and inpainting to offer a robust tool for image editing without the need for extensive dataset training, ensuring adaptability and high-quality results.

## Object Detection and Segmentation
Objects within the image are detected and segmented. Users are then presented with these segments, enabling them to select and remove a specific element from the image.

## User Interaction and Object Masking
Upon user selection, the system generates a detailed mask around the chosen object.

## Object Removal and Image Inpainting
The selected object is masked, and the area is inpainted using the **Deep Image Prior** algorithm. This process synthesizes new pixel data matching the surrounding background, effectively making the removal indistinguishable.

## Hardware

The project was run on NVIDIA GeForce RTX 2080 Ti.

## Create the virtual environment:-

python -m venv nenv       
source nenv/bin/activate 

## Install

pip install torch torchvision matplotlib pillow cv2

## Run the project

python3 main.py


