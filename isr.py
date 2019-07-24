""" ISR """
import numpy as np
from PIL import Image
from ISR.models import RRDN, RDN
from ISR.models import Discriminator
from ISR.models import Cut_VGG19
import glob
from keras import backend as K
import tensorflow as tf

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

import shutil

K.tensorflow_backend._get_available_gpus()

configuration = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
sess = tf.Session(config=configuration) 
K.set_session(sess)

lr_train_patch_size = 40
layers_to_extract = [5, 9]
scale = 8
hr_train_patch_size = lr_train_patch_size * scale

rrdn  = RRDN(arch_params={'C':4, 'D':3, 'G':64, 'G0':64, 'T':10, 'x':scale}, patch_size=lr_train_patch_size)
f_ext = Cut_VGG19(patch_size=hr_train_patch_size, layers_to_extract=layers_to_extract)
discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)

###################################################################################################################

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

###################################################################################################################

def tiled_sr(img, height=128, width=128):
    tiles = get_tile_images(img, height, width)
    sr_rows = []
    for y, row in enumerate(tiles):
        sr_row = []
        for x, tile in enumerate(row):
            sr_tile = rdn.predict(tile)
            sr_row.append(sr_tile)
        sr_rows.append(np.hstack(sr_row))
        
    return np.vstack(sr_rows)

# https://stackoverflow.com/questions/48482317/slice-an-image-into-tiles-using-numpy/48483743
import numpy as np

def get_tile_images(image, height=8, width=8):
    _nrows, _ncols, depth = image.shape
    _size = image.size
    _strides = image.strides

    nrows, _m = divmod(_nrows, height)
    ncols, _n = divmod(_ncols, width)
    if _m != 0 or _n != 0:
        return None

    return np.lib.stride_tricks.as_strided(
        np.ravel(image),
        shape=(nrows, ncols, height, width, depth),
        strides=(height * _strides[0], width * _strides[1], *_strides),
        writeable=False
    )

gen_path = '/home/sandeeppanku/Public/Code/'

# Get the image
inimgPathTrain = '/home/sandeeppanku/Public/Code/superres/data/train/*-in*'
outimgPathTrain = '/home/sandeeppanku/Public/Code/superres/data/train/*-out*'

#train Files
infilenames = sorted({int(file.split("/")[-1].split("-")[0]):file for file in glob.glob(inimgPathTrain)}.items(), key = lambda x:x[0])
outfilenames = sorted({int(file.split("/")[-1].split("-")[0]):file for file in glob.glob(outimgPathTrain)}.items(), key = lambda x:x[0])

#test set
inimgPathTest = '/home/sandeeppanku/Public/Code/superres/data/test/*-in*'
outimgPathTest = '/home/sandeeppanku/Public/Code/superres/data/test/*-out*'

#test Files
infilenamesTest = sorted({int(file.split("/")[-1].split("-")[0]):file for file in glob.glob(inimgPathTest)}.items(), key = lambda x:x[0])
outfilenamesTest = sorted({int(file.split("/")[-1].split("-")[0]):file for file in glob.glob(outimgPathTest)}.items(), key = lambda x:x[0])


# Copy the Files to form the respective image buckets - Train

destlrtrn  = '/home/sandeeppanku/Public/Code/superres/data/trainImageslr/'
desthrtrn  = '/home/sandeeppanku/Public/Code/superres/data/trainImageshr/'
destlrtest = '/home/sandeeppanku/Public/Code/superres/data/testImageslr/'
desthrtest = '/home/sandeeppanku/Public/Code/superres/data/testImageshr/'

if not os.listdir(destlrtrn)  and not os.listdir(desthrtrn) and not os.listdir(desthrtest) and not os.listdir(destlrtest):
	[shutil.copy(f[1], destlrtrn) for f in infilenames]
	[shutil.copy(f[1], desthrtrn) for f in outfilenames]
	[shutil.copy(f[1], destlrtest) for f in infilenamesTest]
	[shutil.copy(f[1], desthrtest) for f in outfilenamesTest]


#assert(len(infilenames) == len(outfilenames))

'''
with Image.open(infilenames[0]) as img:
	img.show()
	lr_img = np.array(img)
	print(f'original shape {lr_img.shape}')

	# Load Model and prediction
	weight_1 = gen_path + 'image-super-resolution/weights/sample_weights/rdn-C6-D20-G64-G064-x2/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5'
	print(f'weight as {weight_1}')
	rdn = RDN(arch_params = {'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
	rdn.model.load_weights('/home/sandeeppanku/Public/Code/image-super-resolution/weights/sample_weights/rdn-C6-D20-G64-G064-x2/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5')


	sr_img = rdn.predict(lr_img)
	#sr_img = tiled_sr(np.array(lr_img), height=300, width=400)
	img = Image.fromarray(sr_img)
	img.show()
'''

##################################################################################################################
# Training


from ISR.train import Trainer
loss_weights = {
  'generator': 0.0,
  'feature_extractor': 0.0833,
  'discriminator': 0.01
}
losses = {
  'generator': 'mae',
  'feature_extractor': 'mse',
  'discriminator': 'binary_crossentropy'
}

log_dirs = {'logs': './logs', 'weights': './weights'}

learning_rate = {'initial_value': 0.0004, 'decay_factor': 0.5, 'decay_frequency': 30}

flatness = {'min': 0.0, 'max': 0.15, 'increase': 0.01, 'increase_frequency': 5}

trainer = Trainer(
    generator=rrdn,
    discriminator=discr,
    feature_extractor=f_ext,
    lr_train_dir='/home/sandeeppanku/Public/Code/superres/data/trainImageslr',
    hr_train_dir='/home/sandeeppanku/Public/Code/superres/data/trainImageshr',
    lr_valid_dir='/home/sandeeppanku/Public/Code/superres/data/testImageslr',
    hr_valid_dir='/home/sandeeppanku/Public/Code/superres/data/testImageshr',
    loss_weights=loss_weights,
    learning_rate=learning_rate,
    flatness=flatness,
    log_dirs=log_dirs,
    weights_generator=None,
    weights_discriminator=None,
    n_validation=40,
)

trainer.train(
    epochs=80,
    steps_per_epoch=500,
    batch_size=16,
    monitored_metrics={'val_PSNR_Y': 'max'}
)
##################################################################################################################