from ultralytics import YOLO
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from IPython.core.magic import register_line_cell_magic
import yaml
from PIL import Image
import os
import seaborn as sns
from ultralytics import YOLO
from matplotlib.patches import Rectangle
import glob
import cv2

 # model load 
model = YOLO("runs/detect/train3/weights/best.pt")

# prediction
results = model.predict(
    source="C:/PPE Detection/dataset/images/train",
    conf=0.005,
    imgsz=640,
    device="cpu",
    show=True,
    save=True,
    stream=True
)
# boxes print 
for r in results:
    print(r.boxes)