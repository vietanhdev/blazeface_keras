import tensorflow as tf
import keras
import numpy as np 

def channel_padding(x):
    """
    zero padding in an axis of channel 
    """

    return keras.backend.concatenate([x, tf.zeros_like(x)], axis=-1)


def single_blaze_block(x, filters=24, kernel_size=5, strides=1, padding='same'):

    # depth-wise separable convolution
    x_0 = keras.layers.SeparableConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False)(x)

    x_1 = keras.layers.BatchNormalization()(x_0) #keras.layers.BatchNormalization

    # Residual connection

    if strides == 2:
        input_channels = x.shape[-1]
        output_channels = x_1.shape[-1]

        x_ = keras.layers.MaxPooling2D()(x)

        if output_channels - input_channels != 0:

            # channel padding
            x_ = keras.layers.Lambda(channel_padding)(x_)

        out = keras.layers.Add()([x_1, x_])
        return keras.layers.Activation("relu")(out)

    out = keras.layers.Add()([x_1, x])
    return keras.layers.Activation("relu")(out)


def double_blaze_block(x, filters_1=24, filters_2=96,
                     kernel_size=5, strides=1, padding='same'):

    # depth-wise separable convolution, project
    x_0 = keras.layers.SeparableConv2D(
        filters=filters_1,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False)(x)

    x_1 = keras.layers.BatchNormalization()(x_0)

    x_2 = keras.layers.Activation("relu")(x_1)

    # depth-wise separable convolution, expand
    x_3 = keras.layers.SeparableConv2D(
        filters=filters_2,
        kernel_size=kernel_size,
        strides=1,
        padding=padding,
        use_bias=False)(x_2)

    x_4 = keras.layers.BatchNormalization()(x_3)

    # Residual connection

    if strides == 2:
        input_channels = x.shape[-1]
        output_channels = x_4.shape[-1]

        x_ = keras.layers.MaxPooling2D()(x)

        if output_channels - input_channels != 0:

            # channel padding
            x_ = keras.layers.Lambda(channel_padding)(x_)

        out = keras.layers.Add()([x_4, x_])
        return keras.layers.Activation("relu")(out)

    out = keras.layers.Add()([x_4, x])
    return keras.layers.Activation("relu")(out)


def BlazeFace(input_shape):

    inputs = keras.layers.Input(shape=input_shape)

    x_0 = keras.layers.Conv2D(
        filters=24, kernel_size=5, strides=2, padding='same')(inputs)
    x_0 = keras.layers.BatchNormalization()(x_0)
    x_0 = keras.layers.Activation("relu")(x_0)

    # Single BlazeBlock phase
    x_1 = single_blaze_block(x_0)
    x_2 = single_blaze_block(x_1)
    x_3 = single_blaze_block(x_2, strides=2, filters=48)
    x_4 = single_blaze_block(x_3, filters=48)
    x_5 = single_blaze_block(x_4, filters=48)

    # Double BlazeBlock phase
    x_6 = double_blaze_block(x_5, strides=2)
    x_7 = double_blaze_block(x_6)
    x_8 = double_blaze_block(x_7)
    x_9 = double_blaze_block(x_8, strides=2)
    x_10 = double_blaze_block(x_9)
    x_11 = double_blaze_block(x_10)

    model = keras.models.Model(inputs=inputs, outputs=[x_8, x_11])
    return model