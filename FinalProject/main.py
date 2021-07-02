# import os

from image_preprocessing import process_images
from model import train
from inference import show_sample
from analysis import analyze_model
from data import get_image_sample

# Modelname: CSRNet
# Based on https://github.com/Neerajj9/CSRNet-keras
# Related paper https://arxiv.org/abs/1802.10062

# Download dataset first:
# https://drive.google.com/file/d/16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI/view

print("Preprocessing images (skipping all files that have already been processed)")
process_images()

print("Show one random image from the dataset including the created groundtruth heatmap")
sample_image = get_image_sample()
show_sample(sample_image)

print("Train the model and save the weights")
train("Model")

print("Show the same sample as before but including the generated heatmap")
show_sample(sample_image, "Model")

print("Analyze the model. Compute the Mean Absolute Error on different dataset subsets.")
analyze_model("Model")
