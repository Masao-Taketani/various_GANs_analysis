import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, LeakyReLU\
    ReLU, Conv2DTranspose, Dropout, concatenate, ZeroPadding2D
from tensorflow.keras.activation import tanh


class InstanceNormalization(Layer):
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


class ResNetBlock(Layer):
    # Define ResNet block

    def __init__(self,
                 filters, 
                 size=3, 
                 strides=1, 
                 padding="VALID", 
                 name="resnetblock"):

        super(ResNetBlock, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.kernel_size = kernel_size
        self.conv2d_1 = Conv2D(filters, 
                               size, 
                               strides, 
                               padding=padding,
                               kernel_initializer=initializer,
                               use_bias=)
        self.instance_norm_1 = InstanceNormalization()
        self.ReLU = ReLU()
        self.conv2d_2 = Conv2D(num_filters,
                               kernel_size,
                               strides,
                               padding=padding,
                               kernel_initializer=initializer,
                               use_bias=)
        self.instance_norm_2 = InstanceNormalization()

        
    def call(inputs):
        pad = int((kernel_size - 1) / 2)
        # Reflection padding was used to reduce artifacts
        x = tf.pad(inputs, [[0, 0], [pad, pad], [pad, pad], [0, 0]], "REFLECT")
        x = self.conv2d_1(x)
        x = self.instance_norm_1(x)
        x = self.ReLU(x)
        x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], "REFLECT")
        x = self.conv2d_2(x)
        x = self.instance_norm_2(x)
        return x + inputs


def get_norm_layer(norm_type):
    if norm_type.lower() == "batchnorm":
        return BatchNormalization()
    elif norm_type.lower() == "instancenorm":
        return InstanceNormalization()
    else:
        raise ValueError("arg `norm_type` has to be either batchnorm "
                         "or instancenorm. What you specified is "
                         "{}".format(norm_type))


def get_activation(activation):
    if activation.lower() == "relu":
        return ReLU()
    elif activation.lower() == "tanh":
        return tanh
    else:
        raise ValueError("arg `norm_type` has to be either relu "
                         "or tanh. What you specified is "
                         "{}".format(norm_type))


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
                 strides=1,
                 padding="same",
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
                             strides=strides,
                             padding=padding,
                             kernel_initializer=initializer,
                             use_bias=False)
        self.relu = ReLU()

    def call(self, inputs):
        x = self.conv2d(inputs)
        if self.norm_type:
            x = self.norm_layer(x)
        x = self.relu(x)

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
                 strides,
                 padding,
                 norm_type="batchnorm",
                 apply_dropout=False,
                 activation="relu",
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
                                               strides=strides,
                                               padding=padding,
                                               kernel_initializer=initializer,
                                               use_bias=False)
        if apply_dropout:
            self.dropout = Dropout(0.5)
        self.activation = get_activation(activation)

    def call(self, inputs):
        x = self.conv2dtranspose(inputs)
        if self.norm_type:
            x = self.norm_layer(x)
        if self.apply_dropout:
            x = self.dropout(x)
        x = self.activation(x)

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



class Pix2Pix(Model):
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
        initializer = tf.random_normal_initializer(0., 0.02)
        self.output_channels = output_channels
        self.norm_type = norm_type
        self.downsample_1 = Downsample(64, 4, norm_type, apply_norm=False)
        self.downsample_2 = Downsample(128, 4, norm_type)
        self.downsample_3 = Downsample(256, 4, norm_type)
        self.downsample_4 = Downsample(512, 4, norm_type)
        self.downsample_5 = Downsample(512, 4, norm_type)
        self.downsample_6 = Downsample(512, 4, norm_type)
        self.downsample_7 = Downsample(512, 4, norm_type)
        self.downsample_8 = Downsample(512, 4, norm_type)

        self.upsample_1 = Upsample(512, 4, norm_type, apply_dropout=True)
        self.upsample_2 = Upsample(512, 4, norm_type, apply_dropout=True)
        self.upsample_3 = Upsample(512, 4, norm_type, apply_dropout=True)
        self.upsample_4 = Upsample(512, 4. norm_type)
        self.upsample_5 = Upsample(256, 4, norm_type)
        self.upsample_6 = Upsample(128, 4, norm_type)
        self.upsample_7 = Upsample(64, 4, norm_type)
        self.conv2d_last = Conv2DTranspose(output_channels,
                                           4,
                                           strides=2,
                                           padding="same",
                                           kernel_initializer=initializer,
                                           activation="tanh")

    def call(self, inputs):
        # input shape: (bs, 256, 256, 3)
        x = self.downsample_1(inputs) # (bs, 128, 128, 64)
        x = self.downsample_2(x) # (bs, 64, 64, 128)
        x = self.downsample_3(x) # (bs, 32, 32, 256)
        x = self.downsample_4(x) # (bs, 16, 16, 512)
        x = self.downsample_5(x) # (bs, 8, 8, 512)
        x = self.downsample_6(x) # (bs, 4, 4, 512)
        x = self.downsample_7(x) # (bs, 2, 2, 512)
        x = self.downsample_8(x) # (bs, 1, 1, 512)

        # refer to https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    

  class ResNetGenerator(Model):
    """
    Referred from CycleGAN paper(https://arxiv.org/abs/1703.10593).
    Let c7s1-k denote a 7×7 Convolution-InstanceNormReLU layer with k filters 
    and stride 1. dk denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with 
    k filters and stride 2. Reflection padding was used to reduce artifacts.
    Rk denotes a residual block that contains two 3 × 3 convolutional layers 
    with the same number of filters on both layer. 
    uk denotes a 3 × 3 fractional-strided-ConvolutionInstanceNorm-ReLU layer 
    with k filters and stride 1/2.

    The network with 6 residual blocks consists of:
        c7s1-64,d128,d256,R256,R256,R256,
        R256,R256,R256,u128,u64,c7s1-3

    The network with 9 residual blocks consists of:
        c7s1-64,d128,d256,R256,R256,R256,
        R256,R256,R256,R256,R256,R256,u128
        u64,c7s1-3

    Args:
    output_channels: number of output channels
          norm_type: normalization type. Either "batchnorm", "instancenorm" or None

    Return:
        Generator model
    """
    def __init__(self,
                 first_filters,
                 output_channels,
                 norm_type="instancenorm", 
                 name="resnet_generator"):

        super(ResNetGenerator, self).__init__(name=name, **kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)
        self.downsample_1 = Downsample(first_filters, 
                                       7,
                                       strides=1,
                                       padding="VALID",
                                       norm_type="instancenorm", 
                                       name="downsample_1",
                                       use_bias=)
        self.downsample_2 = Downsample(first_filters*2, 
                                       3,
                                       strides=2,
                                       norm_type="instancenorm", 
                                       name="downsample_2",
                                       use_bias=))
        self.downsample_3 = Downsample(first_filters*4, 
                                       3,
                                       strides=2,
                                       norm_type="instancenorm", 
                                       name="downsample_2",
                                       use_bias=))
        self.resnetblock_1 = ResNetBlock(first_filters*4, name="resnetblock_1")
        self.resnetblock_2 = ResNetBlock(first_filters*4, name="resnetblock_2")
        self.resnetblock_3 = ResNetBlock(first_filters*4, name="resnetblock_3")
        self.resnetblock_4 = ResNetBlock(first_filters*4, name="resnetblock_4")
        self.resnetblock_5 = ResNetBlock(first_filters*4, name="resnetblock_5")
        self.resnetblock_6 = ResNetBlock(first_filters*4, name="resnetblock_6")
        self.resnetblock_7 = ResNetBlock(first_filters*4, name="resnetblock_7")
        self.resnetblock_8 = ResNetBlock(first_filters*4, name="resnetblock_8")
        self.resnetblock_9 = ResNetBlock(first_filters*4, name="resnetblock_9")
        self.upsample_1 = Upsample(first_filters*2, 
                                   3,
                                   2,
                                   padding="same",
                                   norm_type="intancenorm",
                                   name="upsample_1")
        self.upsample_2 = Upsample(first_filters, 
                                   3,
                                   2,
                                   padding="same",
                                   norm_type="intancenorm",
                                   name="upsample_2")
        self.upsample_3 = Upsample(output_channels, 
                                   7,
                                   1,
                                   padding="VALID",
                                   norm_type=None,
                                   activation="tanh",
                                   name="upsample_3")

    def call(self, inputs):
        # Reflection padding was used to reduce artifacts
        x = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        x = self.downsample_1(x)
        x = self.downsample_2(x)
        x = self.downsample_3(x)
        x = self.resnetblock_1(x)
        x = self.resnetblock_2(x)
        x = self.resnetblock_3(x)
        x = self.resnetblock_4(x)
        x = self.resnetblock_5(x)
        x = self.resnetblock_6(x)
        x = self.resnetblock_7(x)
        x = self.resnetblock_8(x)
        x = self.resnetblock_9(x)
        x = self.upsample_1(x)
        x = self.upsample_2(x)
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        result = self.upsample_3(x)

        return result