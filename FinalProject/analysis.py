import os
import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np

from data import get_image_paths, get_input, get_output, analysis_folder
from model import load_model


def analyze_model(modelname: str, test_a: bool = True, test_b: bool = True, test_ab: bool = True, train_a: bool = True, train_b: bool = True, train_ab: bool = True, test_all: bool = True):
    model = load_model(modelname)
    if test_a:
        images = get_image_paths(False, True, False, False)
        compare_results(model, images, "A_test")
    if test_b:
        images = get_image_paths(False, False, False, True)
        compare_results(model, images, "B_test")
    if train_a:
        images = get_image_paths(True, False, False, False)
        compare_results(model, images, "A_train")
    if train_b:
        images = get_image_paths(False, False, True, False)
        compare_results(model, images, "B_train")
    if test_ab:
        images = get_image_paths(False, True, False, True)
        compare_results(model, images, "AB_test")
    if train_ab:
        images = get_image_paths(True, False, True, False)
        compare_results(model, images, "AB_train")
    if test_all:
        images = get_image_paths()
        compare_results(model, images, "all")


def compare_results(model, images, label: str):
    name = []
    y_true = []
    y_pred = []

    for image in images:
        name.append(image)
        groundtruth, _ = get_output(image)
        num1 = np.sum(groundtruth)
        y_true.append(np.sum(num1))
        img = get_input(image, True)
        num = np.sum(model.predict(img))
        y_pred.append(np.sum(num))

    data = pd.DataFrame({'name': name, 'y_pred': y_pred, 'y_true': y_true})
    mae(y_pred, y_true)
    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)
    data.to_csv('{}/A_on_{}.csv'.format(analysis_folder, label), sep=',')


def mae(y_pred, y_true):
    mae = mean_absolute_error(np.array(y_true), np.array(y_pred))
    print("Mean absolute error: {}".format(mae))
    return mae


if __name__ == "__main__":
    analyze_model("Model")
