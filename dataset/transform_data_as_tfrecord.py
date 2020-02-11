import tensorflow as tf
from tqdm import tqdm
from glob import glob
from PIL import Image
import numpy as np


# img params
img_size = 128
channel_size = 3
dataset_path = "dogs_data/Images/"
tfrecord_output_dir = "dogs_data/tfrecords/"

# discriminator params
input_dims = (img_size, img_size, channel_size)
num_disc_layers = 4
disc_conv_fils = [32, 64, 128, 256]
disc_conv_kernel_size = [3, 3, 3, 3]
disc_conv_strides = [2, 2, 2, 1]
disc_batch_norm_momentum = 0.8
disc_dropout_rate = 0.25

# generator params
z_dims = 100
shape_after_dense = (img_size//4, img_size//4, 128)
gen_upsamp_layers = [True, True, False]
gen_batch_norm_momentum = 0.8
gen_dropout_rate = None
num_gen_layers = 3
gen_conv_fils = [128, 64, channel_size]
gen_conv_kernel_size = [3, 3, 3]

img_paths = []
x_train = []

img_dirs = glob(dataset_path + "*")
for di in img_dirs:
    imgs = glob(di + "/*")
    for img in imgs:
        img_paths.append(img)

print("Start preprocessing data")
for i_path in tqdm(img_paths):
    img = Image.open(i_path)
    resized_img = img.resize((img_size, img_size))
    np_img = np.array(resized_img, dtype=np.float32)
    if np_img.shape != (img_size, img_size, channel_size):
        print("Error found. It will skip the img.")
        print("path:", i_path)
        continue
    np_img = np_img / 127.5 - 1.0
    x_train.append(np_img)
print("Preprocessing is done")

x_train = np.array(x_train)
print('x_train.shape:', x_train.shape)

print('Start Creating TFRcord files for the dogs data.')
def save_data_as_tfrecord(X, tfrecord_filename):
    with tf.python_io.TFRecordWriter(tfrecord_filename) as w:
        for x in tqdm(X):
            x = x.reshape(-1)
            features = tf.train.Features(feature = {
                'X': tf.train.Feature(float_list = tf.train.FloatList(value=x))
            })

            example = tf.train.Example(features=features)
            w.write(example.SerializeToString())

def split_convert_save_data(num_split, X):
    splited_data = np.array_split(X, num_split)
    ct = 0
    for i in range(num_split):
        fname = "dogs-%.2d.tfrecord" % (i)
        splited_x = splited_data[ct]
        save_data_as_tfrecord(splited_x, tfrecord_output_dir + fname)
        ct += 1
    
split_convert_save_data(25, x_train)
print('TFRcord files are created for the dogs data.')
