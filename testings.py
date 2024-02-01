from ultralytics import YOLO
import cv2

from ultralytics import YOLO

# Load a model
model = YOLO('best.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(['1.jpg'])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs