# from ultralytics import YOLO

# # Load pretrained YOLOv8 small model
# model = YOLO("yolov8s.pt")

# # Train the model
# model.train(
#     data="C:/PPE Detection/dataset/data.yaml",
#     epochs=100,
#     imgsz=640,
#     batch=16,
#     device="cpu",  # GPU, set to 0. If CPU, use 'cpu'
#     name="ppe_detection"
# )
# from ultralytics import YOLO

# model = YOLO("yolov8s.pt")

# model.train(
#     data="C:/PPE Detection/dataset/data.yaml",
#     epochs=50,
#     imgsz=640,
#     batch=8,
#     device="cpu",
#     name="ppe_detection_v25"
# )
from ultralytics import YOLO

model = YOLO("runs/detect/ppe_detection_v25/weights/last.pt")

model.train(
    data="C:/PPE Detection/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    resume=True
)