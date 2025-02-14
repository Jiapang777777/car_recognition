import os
# os.environ["OMP_NUM_THREADS"]='2'

from ultralytics import YOLO
# Load a model
model = YOLO('yolo11s-pose.yaml')  # build a new model from YAML
model = YOLO('yolo11s-pose.pt')  # load a pretrained model (recommended for training)  

# Train the model
model.train(data=r'cfg/v11-plate.yaml', epochs=240, imgsz=640, batch=32, device=[0])