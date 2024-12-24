from ultralytics import YOLO
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
# Create a new YOLO model from scratch
# By Nestor model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8s.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="data.yaml", epochs=30, batch=8)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
#results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
success = model.export(format="onnx")
