import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as CM
from PIL import Image

from data import get_input, get_output
from model import load_model


def predict(modelname: str, img_path: str):
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


def show_sample(img_path: str, modelname: str = None):
    '''
    Loads the specified model from the model-directory, loads the specified image.
    Plots the original sample, the groundtruth-heightmap and the predicted-heightmap
    '''
    if modelname is None:
        img = Image.open(img_path)
        img = np.array(img)
        groundtruth, _ = get_output(img_path)
        count = np.sum(groundtruth)
        hmap = None
    else:
        count, img, hmap = predict(modelname, img_path)
        groundtruth, _ = get_output(img_path)
    if groundtruth is None:
        count = int(np.sum(count)) + 1
        print("Prediction: {}".format(count))
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img)
        ax[1].imshow(hmap.reshape(
            hmap.shape[1], hmap.shape[2]), cmap=CM.jet)
        ax[0].set_title("Original image")
        ax[1].set_title("Prediction ({})".format(count))
        
    elif hmap is None:
        count = int(np.sum(count)) + 1
        print("Groundtruth: {}".format(count))
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img)
        ax[1].imshow(groundtruth, cmap=CM.jet)
        ax[0].set_title("Original image")
        ax[1].set_title("Groundtruth ({})".format(count))
    else:
        groundtruth = int(np.sum(groundtruth)) + 1
        count = int(np.sum(count)) + 1
        print("Groundtruth: {}; Prediction: {}".format(groundtruth, count))

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(img.reshape(img.shape[1], img.shape[2], img.shape[3]))
        ax[1].imshow(hmap.reshape(hmap.shape[1], hmap.shape[2]), cmap=CM.jet)
        ax[2].imshow(groundtruth, cmap=CM.jet)
        ax[0].set_title("Original image")
        ax[1].set_title("Prediction ({})".format(count))
        ax[2].set_title("Groundtruth ({})".format(groundtruth))
    plt.show()


if __name__ == "__main__":
    from tkinter.filedialog import askopenfilename

    filename = "y"
    while filename != "None":
        filename = askopenfilename()
        show_sample(filename, "Model")
