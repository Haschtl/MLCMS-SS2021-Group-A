import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np

from data import get_image_paths, load_model, get_input, get_output


def analyze_model(modelname):
    model = load_model(modelname)
    img_paths = get_image_paths()

    name = []
    y_true = []
    y_pred = []

    for image in img_paths:
        name.append(image)
        groundtruth, _ = get_output(image)
        num1 = np.sum(groundtruth)
        y_true.append(np.sum(num1))
        img = get_input(image)
        num = np.sum(model.predict(img))
        y_pred.append(np.sum(num))

    data = pd.DataFrame({'name': name, 'y_pred': y_pred, 'y_true': y_true})
    mae(y_pred,y_true)
    data.to_csv('analysis/A_on_A_test.csv', sep=',')


def mae(y_pred, y_true):
    mae = mean_absolute_error(np.array(y_true), np.array(y_pred))
    print("Mean absolute error: {}".format(mae))
    return mae

if __name__ == "__main__":
    analyze_model("Model")