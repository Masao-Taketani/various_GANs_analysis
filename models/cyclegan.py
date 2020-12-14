import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, LeakyReLU\
    Conv2DTranspose, Dropout, concatenate, ZeroPadding2D


def InstanceNormalization(Layer):
    """
    Instance Normalization Layer (https://arxiv.org/abs/1607.08022).

    Args:
        epsilon: a small positive decimal number to avoid dividing by 0
    """

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


class Downsample(Layer):
    """
     Conv2D -> BatchNorm(or InstanceNorm) -> LeakyReLU

     Args:
        filters: number of filters
           size: filter size
      norm_type: normalization type. Either "batchnorm", "instancenorm" or None
           name: name of the layer

    Return:
        Downsample functional model
    """

    def __init__(self, 
                 filters, 
                 size, 
                 norm_type="batchnorm", 
                 name="downsample", 
                 **kwargs):

        super(Downsample, self).__init__(name=name, **kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)
        self.norm_type = norm_type
        if self.norm_type:
            self.norm_layer = get_norm_layer(norm_type)
        self.conv2d = Conv2D(filters,
                             size,
                             strides=2,
                             padding="same",
                             kernel_initializer=initializer,
                             use_bias=False)

    def call(self, inputs):
        x = self.conv2d(inputs)
        if self.norm_type:
            x = self.norm_layer(x)

        return x


class Upsample(Layer):
    """
    Conv2DTranspose -> BatchNorm(or InstanceNorm) -> Dropout -> ReLU

     Args:
        filters: number of filters
           size: filter size
      norm_type: normalization type. Either "batchnorm", "instancenorm" or None
  apply_dropout: If True, apply the dropout layer
           name: name of the layer

    Return:
        Upsample functional model
    """
    def __init__(self, 
                 filters, 
                 size, 
                 norm_type="batchnorm",
                 apply_dropout=False, 
                 name="upsample", 
                 **kwargs):

        super(Upsample, self).__init__(name=name, **kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)
        self.norm_type = norm_type
        if self.norm_type:
            self.norm_layer = get_norm_layer(norm_type)
        self.apply_dropout = apply_dropout
        self.conv2dtranspose = Conv2DTranspose(filters,
                                               size,
                                               strides=2,
                                               padding="same",
                                               kernel_initializer=initializer,
                                               use_bias=False)
        self.dropout = Dropout(0.5)

    def call(self, inputs):
        x = self.conv2dtranspose(inputs)
        if self.norm_type:
            x = self.norm_layer(x)
        if self.apply_dropout:
            x = self.dropout(x)

        return x


class Discriminator(Model):
    """
    PatchGan discriminator model (https://arxiv.org/abs/1611.07004).

    Args:
        norm_type: normalization type. Either "batchnorm", "instancenorm" or None
           target: Bool, indicating whether the target image is an input or not

    Return:
        Discriminator model
    """
    def __init__(self, 
                 norm_type="batchnorm", 
                 target=True, 
                 name="discriminator"):

        super(Discriminator, self).__init__(name=name, **kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)
        self.norm_type = norm_type
        self.target = target
        self.downsample_1 = Downsample(64, 4, norm_type)
        self.downsample_2 = Downsample(128, 4, norm_type)
        self.downsample_3 = Downsample(256, 4, norm_type)
        self.zeropadding2d_1 = ZeroPadding2D()
        self.conv2d_1 = Conv2D(512, 
                               4, 
                               strides=1, 
                               kernel_initializer=initializer,
                               use_bias=False)
        if self.norm_type:
            self.norm_layer = get_norm_layer(norm_type)
        self.leaky_relu = LeakyReLU()
        self.zeropadding2d_2 = ZeroPadding2D()
        self.conv2d_2 = Conv2D(1, 
                               4, 
                               strides=1, 
                               kernel_initializer=initializer)

    def call():



def get_norm_layer(norm_type):
    if norm_type.lower() == "batchnorm":
        return BatchNormalization()
    elif norm_type.lower() == "instancenorm":
        return InstanceNormalization()
    else:
        raise ValueError("arg `norm_type` has to be either batchnorm "
                            "or instancenorm")


"""
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

    
    if apply_dropout:
        result.add(Dropout(0.5))

     result.add(ReLU())

     return result


def discriminator(norm_type="batchnorm", target=True):
    # PatchGAN discriminator is used
    # target: whether target image is an input or not

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = Input(shape=[None, None, 3], name="disc_input_image")
    x = inp # (bs, 256, 256, channels)

    if target:
        tar = Input(shape=[None, None, 3], name="target_image")
        x = concatenate([inp, tar]) # (bs, 256, 256, channels * 2)
    
    down1 = downsample(64, 4, norm_type, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4, norm_type)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4, norm_type)(down2) # (bs, 32, 32, 256)

    zero_pad1 = ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = Conv2D(512,
                  4,
                  strides=1,
                  kernel_initializer=initializer,
                  use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    norm1 = check_norm_type(norm_type, conv)
    leaky_relu = LeakyReLU()(norm1)
    zero_pad2 = ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)
    last = Conv2D(1,
                  4,
                  strides=1,
                  kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    if target:
        return Model(inputs=[inp, tar], outputs=last)
    else:
        return Model(inputs=inp, outputs=last)
"""


class ResNetGenerator(Model):
    """
    Args:
    output_channels: number of output channels
          norm_type: normalization type. Either "batchnorm", "instancenorm" or None

    Return:
        Generator model
    """
    def __init__(self, 
                 output_channels, 
                 norm_type="batchnorm", 
                 name="resnet_generator"):

        super(ResNetGenerator, self).__init__(name=name, **kwargs)
        self.output_channels = output_channels
        self.norm_type = norm_type