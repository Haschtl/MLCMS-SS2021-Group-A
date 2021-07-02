import os
import h5py
import cv2
import glob
from tensorflow.keras.models import model_from_json
from PIL import Image
import numpy as np
import random


dataset_folder = "data"
model_folder = "models"
weights_folder = "weights"
analysis_folder = "analysis"


def get_image_sample(**kwargs):
    images = get_image_paths(**kwargs)
    return random.choice(images)


def get_image_paths(a_train: bool = True, a_test: bool = True, b_train: bool = True, b_test: bool = True):
    '''
    Loads all pathes for the ShanghaiTech dataset in the datafolder

    Returns
    -------
    list
        list of pathes to all images  
    '''

    path_sets = []
    if a_train:
        path_sets.append(os.path.join(
            dataset_folder, 'part_A_final/train_data', 'images'))
    if a_test:
        path_sets.append(os.path.join(
            dataset_folder, 'part_A_final/test_data', 'images'))
    if b_train:
        path_sets.append(os.path.join(
            dataset_folder, 'part_B_final/train_data', 'images'))
    if b_test:
        path_sets.append(os.path.join(
            dataset_folder, 'part_B_final/test_data', 'images'))
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    return img_paths


def imagepath2groundpath(path: str):
    '''
    Loads an image-path and returns the corresponding groundtruth path

    Returns
    -------
    filepath
        Filepath to groundtruth data
    '''
    return path.replace('.jpg', '.h5').replace('images', 'ground')


def groundpath2imagepath(path: str):
    '''
    Loads a groundtruth-path and returns the corresponding image-path 

    Returns
    -------
    filepath
        Filepath to image data
    '''
    return path.replace('.h5', '.jpg').replace('ground', 'images')


def get_input(path: str, expand=False):
    '''
    Loads an image, scales it to [0,1] and applies a color-filter

    Returns
    -------
    image
        original image data
    '''
    path = groundpath2imagepath(path)
    im = Image.open(path).convert('RGB')

    im = np.array(im)

    im = im/255.0

    im = filter_input(im)

    if expand:
        im = np.expand_dims(im, axis=0)
    return im


def filter_input(image):

    image[:, :, 0] = (image[:, :, 0]-0.485)/0.229
    image[:, :, 1] = (image[:, :, 1]-0.456)/0.224
    image[:, :, 2] = (image[:, :, 2]-0.406)/0.225
    return image


def get_output(path: str):
    '''
    Loads groundtruth data

    Returns
    -------
    array
        density array
    img
        scaled image for model
    '''
    path = imagepath2groundpath(path)
    try:
        gt_file = h5py.File(path, 'r')
        target = np.asarray(gt_file['density'])
        img = scale_image(target, 8)
        img = np.expand_dims(img, axis=2)

        return target, img
    except FileNotFoundError:
        return None, None


def scale_image(image, factor=8, inv=False):
    if inv:
        factor = 1/factor
    return cv2.resize(image, (int(
        image.shape[1]/factor), int(image.shape[0]/factor)), interpolation=cv2.INTER_CUBIC)*factor*factor


def write_output(density, file_path: str):
    '''
    Writes a density-map to the output
    '''
    file_path = imagepath2groundpath(file_path)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with h5py.File(file_path, 'w') as hf:
        hf['density'] = density
