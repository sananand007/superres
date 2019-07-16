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
#TODO 
New Metrics to be added

epoch
9
examples

graph

image_psnr - toadd
20.349

loss
0.06469

lr - toadd
0.00005

perceptual_distance
53.628

val_image_psnr --to add
19.755

val_loss
0.06835

val_perceptual_distance
56.453

Chnages 
'''

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

config.num_epochs = 3
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


custom_objects={'perceptual_distance': perceptual_distance, 'image_psnr':image_psnr}
modelFile = '/home/sandeeppanku/Public/Code/superres/models/modelcl.h5'
#model = model_from_json(open(modelFile).read())
#model.load_weights(os.path.join(os.path.dirname(modelFile), modelFile))

'''
Golden Model
model = load_model(modelFile, custom_objects=custom_objects)
'''
#print(f'Shape of model {model.summary()}')



model = Sequential()
model.add(Conv2D(3, (3, 3), activation='relu', padding='same',
                        input_shape=(config.input_width, config.input_height, 3)))
model.add(UpSampling2D())
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D())
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D())
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))


# DONT ALTER metrics=[perceptual_distance]
model.compile(optimizer='adam', loss='mse',
              metrics=[perceptual_distance, image_psnr])

model.fit_generator(image_generator(config.batch_size, train_dir),
                    steps_per_epoch=config.steps_per_epoch,
                    epochs=config.num_epochs, callbacks=[
                        ImageLogger(), WandbCallback()],
                    validation_steps=config.val_steps_per_epoch,
                    validation_data=val_generator)

#model.save(os.path.join(wandb.run.dir, "model_hell1.h5"))

