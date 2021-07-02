import os
import math
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable gpu
import sys
import numpy as np
import random
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard


from data import get_image_paths, get_input, get_output, model_folder, weights_folder


defaultOptions = {
    "optimizer": "SGD",
    "optimizer_options": {
        "learning_rate": 1e-7,
        "decay": 5*1e-4,
        "momentum": 0.95,
    },
    "lossfunction": "euclidean-norm",
    "train_options": {
        "batch_size": 64,   # 1
        "epochs": 5,  # 1
        "steps_per_epoch": 10   # 700
    },
    "model_options": {
        "batch_norm": False
    }
}


def train(modelname: str = "Model", options: dict = None, tensorboard=False):
    '''
    Main training function for the model.
    Specify a name for the model (required for filenames).
    Specify options like the defaultOptions above.
    '''
    if options is None:
        options = defaultOptions
    else:
        options = {**defaultOptions, **options}

    img_paths = get_image_paths(True, False, True, False)
    model = CrowdNet(options["model_options"], True)
    model.summary()

    train_gen = image_generator(
        img_paths, options["train_options"]["batch_size"])

    optimizer_options = options["optimizer_options"]
    if options["optimizer"] == "SGD":
        optimizer = SGD(learning_rate=optimizer_options["learning_rate"], decay=(
            optimizer_options["decay"]), momentum=optimizer_options["momentum"])
    else:
        print("Only 'SGD' Optimizer is supported")
        sys.exit(-1)

    if options["lossfunction"] == "euclidean-norm":
        loss = euclidean_distance_loss
    else:
        print("Only 'euclidean-norm' Loss is supported")
        sys.exit(-1)

    model.compile(optimizer=optimizer,
                  loss=loss, metrics=['mse'])

    if tensorboard:
        callbacks = [TensorBoard(log_dir="log", histogram_freq=1)]
    else:
        callbacks = None
    train_options = options["train_options"]

    model.fit(
        train_gen, epochs=train_options["epochs"], steps_per_epoch=train_options["steps_per_epoch"], verbose=2, callbacks=callbacks)
    save_model(model, modelname)


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
    # with open(os.path.join(model_folder, filename+".json"), 'r') as json_file:
    #     loaded_model_json = json_file.read()
    # loaded_model = model_from_json(loaded_model_json, custom_objects={'CrowdNet': CrowdNet})
    loaded_model = CrowdNet()
    loaded_model.load_weights(os.path.join(weights_folder, weightsname))
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



def image_generator(files, batch_size: int = 64):
    '''
    Creates batches for model-training
    (necessary as tensorflow only allows batches of data with same shape but images have different shapes)
    '''
    while True:
        batch_paths = np.random.choice(a=files, size=batch_size)

        batch_input = []
        batch_output = []
        for input_path in batch_paths:
            # input(input_path)
            _input = get_input(input_path)
            _, _output = get_output(input_path)
            batch_input.append(_input)
            batch_output.append(_output)
        batch_input, batch_output = crop_images(batch_input, batch_output)
        batch_x = np.stack(batch_input)
        batch_y = np.stack(batch_output)

        yield(batch_x, batch_y)


def crop_images(input, output):
    '''
    Crop the image and the target heatmap (also resizes the target)
    '''
    minshape_out = get_min_shape(output)
    minshape_in = (minshape_out[0]*8, minshape_out[1]*8)
    cropped_input = []
    cropped_output = []
    for idx,_ in enumerate(output):
        cropped_output.append(crop_center(
            output[idx], minshape_out[0], minshape_out[1]))
        cropped_input.append(crop_center(
            input[idx], minshape_in[0], minshape_in[1]))
    return cropped_input, cropped_output


def crop_center(img, crop_x, crop_y):
    start_x = math.floor(img.shape[0]/2-(crop_x/2))
    start_y = math.floor(img.shape[1]/2-(crop_y/2))
    return img[start_x:start_x+crop_x, start_y:start_y+crop_y,:]


def get_min_shape(arrays):
    x = [a.shape[0] for a in arrays]
    y = [a.shape[1] for a in arrays]
    return (min(x), min(y))


def euclidean_distance_loss(y_true, y_pred):
    '''
    Euclidean distance as a measure of loss (Loss function)
    '''
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


# Neural network model : VGG + Conv
class CrowdNet(Sequential):
    def __init__(self, options: dict = None, init_weights: bool = False, **kwargs):
        super(CrowdNet, self).__init__(**kwargs)
        if options is None:
            options = defaultOptions["model_options"]
        # Batch Normalisation option
        self.batch_norm = options["batch_norm"]
        self.vgg_kernel_size = (3, 3)
        if self.batch_norm:
            self.kernel_initializer = 'glorot_uniform'
        else:
            self.kernel_initializer = RandomNormal(stddev=0.01)

        self.addVGG16Layers()
        self.addGAPLayers()
        if init_weights:
            self.init_weights("VGG_16")

    def addVGG16Layers(self):
        # VGG-16 layers - do not touch! Pretrained weights!
        # Variable Input Size
        self.addVGGBlock(2, 64, True, (None, None, 3))
        self.addVGGBlock(2, 128, True)
        self.addVGGBlock(3, 256, True)
        self.addVGGBlock(3, 512, False)

    def addVGGBlock(self, convolutions: int = 2, filters: int = 256, maxpool: bool = True, input_shape=None):
        for conv in range(convolutions):
            if input_shape is not None:
                self.add(Conv2D(filters, input_shape=input_shape, kernel_size=self.vgg_kernel_size, kernel_initializer=self.kernel_initializer,
                                activation='relu', padding='same', name="VGG_C{}_{}".format(filters, conv)))
            else:
                self.add(Conv2D(filters, kernel_size=self.vgg_kernel_size, kernel_initializer=self.kernel_initializer,
                                activation='relu', padding='same', name="VGG_C{}_{}".format(filters, conv)))
            if self.batch_norm:
                self.add(BatchNormalization())
            input_shape = None

        if maxpool:
            self.add(MaxPooling2D(strides=2, name="VGG_MAX_{}".format(filters)))

    def addGAPLayers(self):
        # Conv2D
        self.add(Conv2D(512, (3, 3), activation='relu',
                        dilation_rate=2, kernel_initializer=self.kernel_initializer, padding='same', name="GAP_C512_0"))
        self.add(Conv2D(512, (3, 3), activation='relu',
                        dilation_rate=2, kernel_initializer=self.kernel_initializer, padding='same', name="GAP_C512_1"))
        self.add(Conv2D(512, (3, 3), activation='relu',
                        dilation_rate=2, kernel_initializer=self.kernel_initializer, padding='same', name="GAP_C512_2"))
        self.add(Conv2D(256, (3, 3), activation='relu',
                        dilation_rate=2, kernel_initializer=self.kernel_initializer, padding='same', name="GAP_C256_0"))
        self.add(Conv2D(128, (3, 3), activation='relu',
                        dilation_rate=2, kernel_initializer=self.kernel_initializer, padding='same', name="GAP_C128_0"))
        self.add(Conv2D(64, (3, 3), activation='relu',
                        dilation_rate=2, kernel_initializer=self.kernel_initializer, padding='same', name="GAP_C64_0"))
        self.add(Conv2D(1, (1, 1), activation='relu',
                        dilation_rate=1, kernel_initializer=self.kernel_initializer, padding='same', name="GAP_C1_0"))

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
            if('VGG_C' in self.layers[i+offset].name):
                self.layers[i+offset].set_weights(vgg_weights[i])
                i = i+1
                # print('h')

            else:
                offset = offset+1


if __name__ == "__main__":
    train()
