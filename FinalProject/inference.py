import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as CM
from PIL import Image

from data import get_input, get_output
from model import load_model


def predict(modelname:str, img_path:str):
    '''
    Loads the specified model from the model-directory, loads the specified image.
    Predicts the amount of objects in the image.

    Returns
    -------
    count
        Number of objects in the image
    image
        The input image
    hmap
        The predicted heightmap  
    '''
    # Function to load image,predict heat map, generate count and return (count , image , heat map)
    model = load_model(modelname)
    image = get_input(img_path, True)
    hmap = model.predict(image)
    count = np.sum(hmap)
    return count, image, hmap


def show_sample(img_path: str, modelname: str=None):
    '''
    Loads the specified model from the model-directory, loads the specified image.
    Plots the original sample, the groundtruth-heightmap and the predicted-heightmap
    '''
    if modelname is None:
        img = Image.open(img_path)
        img = np.array(img)
        groundtruth, _ = get_output(img_path)
        count = np.sum(groundtruth)
        hmap=None 
    else:
        count, img, hmap = predict(modelname, img_path)
        groundtruth, _ = get_output(img_path)
    if groundtruth is None or hmap is None:
        if hmap is None:
            print("Groundtruth: {}".format(int(np.sum(count)) + 1))
        else:
            print("Prediction: {}".format(int(np.sum(count)) + 1))
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img)
        if hmap is None:
            ax[1].imshow(groundtruth, cmap=CM.jet)
        else:
            ax[1].imshow(hmap.reshape(hmap.shape[1], hmap.shape[2]), cmap=CM.jet)

    else:
        print("Groundtruth: {}; Prediction: {}".format(
            int(np.sum(groundtruth)) + 1, int(np.sum(count)) + 1))

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(img.reshape(img.shape[1], img.shape[2], img.shape[3]))
        ax[1].imshow(hmap.reshape(hmap.shape[1], hmap.shape[2]), cmap=CM.jet)
        ax[2].imshow(groundtruth, cmap=CM.jet)
    plt.show()
