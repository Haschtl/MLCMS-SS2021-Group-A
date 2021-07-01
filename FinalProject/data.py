import os
import h5py
import cv2
import glob
from tensorflow.keras.models import model_from_json
from PIL import Image
import numpy as np

dataset_folder = "data"
model_folder = "models"
weights_folder = "weights"


def get_image_paths():
    '''
    Loads all pathes for the ShanghaiTech dataset in the datafolder

    Returns
    -------
    list
        list of pathes to all images  
    '''
    part_a_train = os.path.join(
        dataset_folder, 'part_A_final/train_data', 'images')
    part_a_test = os.path.join(
        dataset_folder, 'part_A_final/test_data', 'images')
    part_b_train = os.path.join(
        dataset_folder, 'part_B_final/train_data', 'images')
    part_b_test = os.path.join(
        dataset_folder, 'part_B_final/test_data', 'images')
    path_sets = [part_a_train, part_a_test, part_b_train, part_b_test]
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    return img_paths


def load_model(filename: str = 'Model', weightsname: str = None):
    '''
    Loads the specified model from the model-directory including their corresponding weights.
    Do not specify filetypes here! You can also specify the weights as you want

    Returns
    -------
    model
        Tensorflow model
    '''
    if weightsname is None:
        weightsname = filename+".h5"
    json_file = open(os.path.join(model_folder, filename+".json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join(weights_folder, weightsname+".h5"))
    return loaded_model


def save_model(model, filename: str, weightsname: str = None):
    '''
    Saves the model to the model-directory including their corresponding weights.
    Do not specify filetypes here! You can also specify the weights-filename as you want.
    '''
    if weightsname is None:
        weightsname = filename+".h5"
    model.save_weights(os.path.join(weights_folder, weightsname))
    model_json = model.to_json()
    with open(os.path.join(model_folder, filename+".json"), "w") as json_file:
        json_file.write(model_json)


def imagepath2groundpath(path:str):
    '''
    Loads an image-path and returns the corresponding groundtruth path

    Returns
    -------
    filepath
        Filepath to groundtruth data
    '''
    return path.replace('.jpg', '.h5').replace('images', 'ground')


def groundpath2imagepath(path:str):
    '''
    Loads a groundtruth-path and returns the corresponding image-path 

    Returns
    -------
    filepath
        Filepath to image data
    '''
    return path.replace('.h5', '.jpg').replace('ground', 'images')


def get_input(path:str):
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

    im[:, :, 0] = (im[:, :, 0]-0.485)/0.229
    im[:, :, 1] = (im[:, :, 1]-0.456)/0.224
    im[:, :, 2] = (im[:, :, 2]-0.406)/0.225

    im = np.expand_dims(im, axis=0)
    return im


def get_output(path:str):
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
        img = cv2.resize(target, (int(
            target.shape[1]/8), int(target.shape[0]/8)), interpolation=cv2.INTER_CUBIC)*64
        img = np.expand_dims(img, axis=3)

        return target, img
    except Exception:
        return None, None


def write_output(density, file_path:str):
    '''
    Writes a density-map to the output
    '''
    file_path = imagepath2groundpath(file_path)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with h5py.File(file_path, 'w') as hf:
        hf['density'] = density
