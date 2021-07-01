import os

from image_preprocessing import process_images
from model import train
from inference import show_sample
from analysis import analyze_model

# Based on https://github.com/Neerajj9/CSRNet-keras

# Download dataset first:
# https://drive.google.com/file/d/16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI/view

if not os.path.exists("data/part_A_final/train_data/ground"):
    print("Images have not been preprocessed yet... doing this now.")
    process_images()

show_sample("data/part_A_final/train_data/images/IMG_101.jpg")

train("Model")

show_sample("data/part_A_final/train_data/images/IMG_101.jpg", "Model")

analyze_model("Model")
