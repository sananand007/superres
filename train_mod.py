
import cv2
import glob
import numpy as np
from keras import regularizers
from keras.models import load_model, Model
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, Input, Lambda, add
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam, RMSprop
#rom utilities import random_crop, ssim, content_fn, test_edsr, ImageLoader
import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import wandb
from wandb.keras import WandbCallback

'''
Mods
'''
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, AveragePooling3D, Conv3D, MaxPooling3D, TimeDistributed, MaxPooling3D, UpSampling3D
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Dropout
from keras.layers import Dense
from keras.optimizers import SGD
import random
import glob
import wandb
from wandb.keras import WandbCallback
import subprocess
import os
from PIL import Image
import numpy as np
from keras import backend as K
from keras.layers import GRU, LSTM, ConvLSTM2DCell
from keras.models import load_model, model_from_json
import cv2

# Added for CNN model
#from keras.layers.convolutional import Conv3D

from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dense, Reshape
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import InputLayer
from keras.models import Model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.optimizers import Adagrad

## Adding a vgg16
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.models import Model

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


K.tensorflow_backend._get_available_gpus()

configuration = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
sess = tf.Session(config=configuration) 
K.set_session(sess)

##########################################################################

#run = wandb.init(project='superres')
run = wandb.init(project='respick')
config = run.config

config.num_epochs = 50
config.batch_size = 32
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256

val_dir = 'data/test'
train_dir = 'data/train'

# automatically get the data if it doesn't exist
if not os.path.exists("data"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "mkdir data && curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xz -C data", shell=True)

config.steps_per_epoch = len(
    glob.glob(train_dir + "/*-in.jpg")) // config.batch_size
config.val_steps_per_epoch = len(
    glob.glob(val_dir + "/*-in.jpg")) // config.batch_size


def image_generator(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        random.shuffle(input_filenames)
        if counter+batch_size >= len(input_filenames):
            counter = 0
        for i in range(batch_size):
            img = input_filenames[counter + i]
            small_images[i] = np.array(Image.open(img)) / 255.0
            large_images[i] = np.array(
                Image.open(img.replace("-in.jpg", "-out.jpg"))) / 255.0
        yield (small_images, large_images)
        counter += batch_size


def perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    print("Inside the perceptual function", y_true.shape, y_pred.shape)
    y_true *= 255
    y_pred *= 255
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


val_generator = image_generator(config.batch_size, val_dir)
in_sample_images, out_sample_images = next(val_generator)

print(f'size of images input image: {in_sample_images.shape}, output image: {out_sample_images.shape}')


def image_psnr(im1, im2):
    """ Calculate the image psnr """
    print(f'image 1 shape {im1.shape}, image 2 shape {im2.shape}')
    #img_arr1 = np.array(im1).astype('float32')
    #img_arr2 = np.array(im2).astype('float32')
    #mse = tf.reduce_mean(tf.squared_difference(img_arr1, img_arr2))
    #psnr = tf.constant(255**2, dtype=tf.float32)/mse
    # result = tf.constant(10, dtype=tf.float32)*log10(psnr)
    # with tf.Session():
    #     result = result.eval()
    im1 = tf.image.convert_image_dtype(im1, tf.float32)
    im2 = tf.image.convert_image_dtype(im2, tf.float32)
    psnr = tf.image.psnr(im1, im2, max_val=1.0)
    return psnr


class ImageLogger(Callback):
    def on_epoch_end(self, epoch, logs):
        preds = self.model.predict(in_sample_images)
        in_resized = []
        for arr in in_sample_images:
            # Simple upsampling
            in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
        wandb.log({
            "examples": [wandb.Image(np.concatenate([in_resized[i] * 255, o * 255, out_sample_images[i] * 255], axis=1)) for i, o in enumerate(preds)]
        }, commit=False)


'''
Residual Dense Network for Image Super-Resolution
'''
custom_objects={'perceptual_distance': perceptual_distance, 'image_psnr':image_psnr}
modelFile = '/home/sandeeppanku/Public/Code/superres/models/modelcl.h5'

model = load_model(modelFile, custom_objects=custom_objects)

# Pop out all the un-necessary layers
for i in range(16):
	model.layers.pop()
Upsample1  = UpSampling2D(name='upsamplingNew1')(model.layers[-1].output)
conv2d_142 = Conv2D(64, (3,3), padding='same', activation='relu', name='conv_new1')(Upsample1)
Upsample2  = UpSampling2D(name='upsamplingNew2')(conv2d_142)
conv2d_143 = Conv2D(64, (3,3), padding='same', activation='relu', name='conv_new2')(Upsample2)
Upsample3  = UpSampling2D(name='upsamplingNew3')(conv2d_143)
conv2d_144 = Conv2D(3, (3,3), padding='same', activation='relu',  name='conv_new3')(Upsample3)


## Adding resnet pre-trained layers
img_shape = (256,256,3)
resnetmdl = ResNet50(include_top=False, weights=None, input_tensor=conv2d_144, input_shape=img_shape, classes=None)

newModel = Model(inputs=model.inputs, outputs = resnetmdl)

print(f'Model is {newModel.summary()}')

'''
Split and Merge Model -- Try later
'''
#model 	= Sequential()
# input1 	= Input(shape=(config.input_width, config.input_height, 3))
# input2 	= Input(shape=(config.input_width, config.input_height, 3))

# conv1 	= Conv2D(64, (3, 3), padding='same')(input1)
# activ1 	= Activation('relu')(conv1)
# conv2 	= Conv2D(64, (3, 3), padding='same')(activ1)
# activ2 	= Activation('relu')(conv2)
# conv3 	= Conv2D(256, (3, 3), padding='same')(activ2)
# activ3 	= Activation('relu')(conv3)
# conv4 	= Conv2D(64, (3, 3), padding='same')(activ3)
# activ4 	= Activation('relu')(conv4)
# merge1 	= concatenate([activ2, activ4])
# model 	= Model(inputs=[input1, input2])
#print(f'model summary {model.summary()}')

'''
Ways to add the modify a saved model
'''

# newModel = Model(inputs=model.inputs, outputs = model.layers[-1].output)
# newModel.set_weights(model.get_weights())
# weights_bak = model.layers[-1].get_weights()
# nb_classes = model.layers[-1].output_shape[-1]
#model.layers[-1].outbound_nodes = []
#model.outputs = [model.layers[-1].output]
#newModel.add(model)
#newModel.set_weights(model.get_weights())
#newModel.add(Conv2D(3, (3, 3), activation='relu', padding='same'))


# DONT ALTER metrics=[perceptual_distance]
newModel.compile(optimizer='adam', loss='mse',
              metrics=[perceptual_distance, image_psnr])

newModel.fit_generator(image_generator(config.batch_size, train_dir),
                    steps_per_epoch=config.steps_per_epoch,
                    epochs=config.num_epochs, callbacks=[
                        ImageLogger(), WandbCallback()],
                    validation_steps=config.val_steps_per_epoch,
                    validation_data=val_generator)