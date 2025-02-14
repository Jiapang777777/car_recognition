import os
# os.environ["OMP_NUM_THREADS"]='2'

from ultralytics import YOLO
# Load a model
model = YOLO('yolo11s.yaml')  # build a new model from YAML
model = YOLO('yolo11s.pt')  # load a pretrained model (recommended for training)  

# Train the model
model.train(data=r'cfg/coco.yaml', epochs=120, imgsz=960, batch=16, device=[0])