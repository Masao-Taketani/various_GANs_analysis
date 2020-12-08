import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, LeakyReLU\
    Conv2DTranspose, Dropout


def InstanceNormalization(Layer):

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(name="scale",
                                     shape=input_shape[-1:],
                                     initializer=tf.random_normal_initializer=(1, 0.02),
                                     trainable=True)

        self.offset = self.add_weight(name="offset",
                                      shape=input_shape[-1:],
                                      initializer="zeros",
                                      trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized * self.offset


def downsample(filters, size, norm_type="batchnorm", apply_norm=True):
    # Conv2D -> BatchNorm(or InstanceNorm) -> LeakyReLU

    initializer = tf.random_normal_initializer(0., 0.02)

    result = Sequential()
    reault.add(Conv2D(filters,
                      size,
                      strides=2,
                      padding="same",
                      kernel_initializer=initializer,
                      use_bias=False))

    if apply_norm:
        if norm_type.lower() == "batchnorm":
            result.add(BatchNormalization())
        elif norm_type.lower() == "instancenorm":
            result.add(InstanceNormalization())
        else:
            raise ValueError("arg `apply_norm` has to be either batchnorm or instancenorm")

    result.add(LeakyReLU())

    return result


def upsample(filters, size, norm_type="batchnorm", apply_dropout=False):
    # Conv2DTranspose -> BatchNorm(or InstanceNorm) -> Dropout -> ReLU
    initializer = tf.random_normal_initializer(0., 0.02)

    result = Sequential()
    result.add(Conv2DTranspose(filters,
                               size,
                               strides=2,
                               padding="same",
                               kernel_initializer=initializer,
                               use_bias=False))

    if norm_type.lower() == "batchnorm":
        result.add(BatchNormalization())
    elif norm_type.lower() == "instancenorm":
        result.add(InstanceNormalization())
    else:
        raise ValueError("arg `apply_nrom` has to be either batchnorm or instancenorm")

    if apply_dropout:
        result.add(Dropout(0.5))

     result.add(ReLU())

     return result


def generator_resnet(img, options, )