import os
import sys
import numpy as np
import random
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K


from data import get_image_paths, get_input, get_output, save_model, model_folder, weights_folder


defaultOptions = {
    "optimizer": "SGD",
    "optimizer_options": {
        "learning_rate": 1e-7,
        "decay": 5*1e-4,
        "momentum": 0.95,
    },
    "lossfunction": "euclidean-distance",
    "train_options": {
        "batch_size": None,
        "epochs": 1,
        "steps_per_epoch": 700
    },
    "model_options": {
        "batch_norm": False,
        "kernel": (3, 3)
    }
}


def train(modelname="Model", options=None):
    if options is None:
        options = defaultOptions
    else:
        options = {**defaultOptions, **options}

    img_paths = get_image_paths()
    model = CrowdNet(options["model_options"])
    model.summary()

    train_gen = image_generator(img_paths, 1)

    optimizer_options = options["optimizer_options"]
    if options["optimizer"] == "SGD":
        optimizer = SGD(lr=optimizer_options["learning_rate"], decay=(
            optimizer_options["decay"]), momentum=optimizer_options["momentum"])
    else:
        print("Only 'SGD' Optimizer is supported")
        sys.exit(-1)
    model.compile(optimizer=optimizer,
                  loss=euclidean_distance_loss, metrics=['mse'])

    train_options = options["train_options"]
    model.fit_generator(
        train_gen, epochs=train_options["epochs"], steps_per_epoch=train_options["steps_per_epoch"], verbose=1)
    save_model(model, modelname)


def preprocess_input(image, target):
    # crop image
    # crop target
    # resize target
    crop_size = (int(image.shape[0]/2), int(image.shape[1]/2))

    if random.randint(0, 9) <= -1:
        dx = int(random.randint(0, 1)*image.shape[0]*1./2)
        dy = int(random.randint(0, 1)*image.shape[1]*1./2)
    else:
        dx = int(random.random()*image.shape[0]*1./2)
        dy = int(random.random()*image.shape[1]*1./2)

    #print(crop_size , dx , dy)
    img = image[dx: crop_size[0]+dx, dy:crop_size[1]+dy]

    target_aug = target[dx:crop_size[0]+dx, dy:crop_size[1]+dy]
    # print(img.shape)

    return(img, target_aug)


def image_generator(files, batch_size=64):
    while True:
        input_path = np.random.choice(a=files, size=batch_size)

        batch_input = []
        batch_output = []

        # for input_path in batch_paths:
        _input = get_input(input_path[0])
        _, _output = get_output(input_path[0])

        batch_input += [_input]
        batch_output += [_output]

        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield(batch_x, batch_y)


def euclidean_distance_loss(y_true, y_pred):
    # Euclidean distance as a measure of loss (Loss function)
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


# Neural network model : VGG + Conv
class CrowdNet(Sequential):
    def __init__(self, options):
        super(CrowdNet, self).__init__()
        # Variable Input Size
        rows = None
        cols = None

        # Batch Normalisation option
        batch_norm = options["batch_norm"]
        kernel = options["kernel"]
        if batch_norm:
            kernel_initializer = 'glorot_uniform'
        else:
            kernel_initializer = RandomNormal(stddev=0.01)

        self.addBatchNorm(Conv2D(64, kernel_size=kernel, input_shape=(
            rows, cols, 3), activation='relu', padding='same'), batch_norm)

        self.addBatchNorm(Conv2D(64, kernel_size=kernel,
                                 activation='relu', padding='same'), batch_norm)

        self.add(MaxPooling2D(strides=2))
        self.addBatchNorm(Conv2D(128, kernel_size=kernel,
                                 activation='relu', padding='same'), batch_norm)

        self.addBatchNorm(Conv2D(128, kernel_size=kernel,
                                 activation='relu', padding='same'), batch_norm)

        self.add(MaxPooling2D(strides=2))
        self.addBatchNorm(Conv2D(256, kernel_size=kernel,
                                 activation='relu', padding='same'), batch_norm)

        self.addBatchNorm(Conv2D(256, kernel_size=kernel,
                                 activation='relu', padding='same'), batch_norm)

        self.addBatchNorm(Conv2D(256, kernel_size=kernel,
                                 activation='relu', padding='same'), batch_norm)

        self.add(MaxPooling2D(strides=2))
        self.addBatchNorm(Conv2D(512, kernel_size=kernel,
                                 activation='relu', padding='same'), batch_norm)

        self.addBatchNorm(Conv2D(512, kernel_size=kernel,
                                 activation='relu', padding='same'), batch_norm)

        self.addBatchNorm(Conv2D(512, kernel_size=kernel,
                                 activation='relu', padding='same'), batch_norm)

        # Conv2D
        self.add(Conv2D(512, (3, 3), activation='relu',
                        dilation_rate=2, kernel_initializer=kernel_initializer, padding='same'))
        self.add(Conv2D(512, (3, 3), activation='relu',
                        dilation_rate=2, kernel_initializer=kernel_initializer, padding='same'))
        self.add(Conv2D(512, (3, 3), activation='relu',
                        dilation_rate=2, kernel_initializer=kernel_initializer, padding='same'))
        self.add(Conv2D(256, (3, 3), activation='relu',
                        dilation_rate=2, kernel_initializer=kernel_initializer, padding='same'))
        self.add(Conv2D(128, (3, 3), activation='relu',
                        dilation_rate=2, kernel_initializer=kernel_initializer, padding='same'))
        self.add(Conv2D(64, (3, 3), activation='relu',
                        dilation_rate=2, kernel_initializer=kernel_initializer, padding='same'))
        self.add(Conv2D(1, (1, 1), activation='relu',
                        dilation_rate=1, kernel_initializer=kernel_initializer, padding='same'))

        self.init_weights("VGG_16")

    def init_weights(self, initmodel: str = "VGG_16"):
        #vgg =  VGG16(weights='imagenet', include_top=False)

        with open(os.path.join(model_folder, initmodel+'.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(os.path.join(
            weights_folder, initmodel+'.h5'))

        vgg = loaded_model

        vgg_weights = []
        for layer in vgg.layers:
            if('conv' in layer.name):
                vgg_weights.append(layer.get_weights())

        offset = 0
        i = 0
        while(i < 10):
            if('conv' in self.layers[i+offset].name):
                self.layers[i+offset].set_weights(vgg_weights[i])
                i = i+1
                # print('h')

            else:
                offset = offset+1

    def addBatchNorm(self, layer, batch_norm=False):
        self.add(layer)
        if batch_norm:
            self.add(BatchNormalization())


if __name__ == "__main__":
    train()
