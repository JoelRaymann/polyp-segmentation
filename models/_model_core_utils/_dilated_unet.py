"""
Scripts to implement the dilated U-Net model
"""
import tensorflow as tf
import layers


# Private functions
def __dilated_identity_block(input_tensor, kernel_size, filters, stage, block, dilation_rate=(2, 2)):
    """The identity dilated block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = tf.keras.layers.Conv2D(filters1, (1, 1),
                               kernel_initializer='he_normal',
                               name=conv_name_base + '2a')(input_tensor)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters2,
                               kernel_size,
                               padding='same',
                               dilation_rate=dilation_rate,
                               kernel_initializer='he_normal',name=conv_name_base + '2b')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters3, (1, 1),
                               kernel_initializer='he_normal',
                               name=conv_name_base + '2c')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = tf.keras.layers.add([x, input_tensor])
    x = tf.keras.layers.Activation('relu', name=conv_name_base + "_add_activation")(x)
    return x


def __dilated_conv_block(input_tensor,
                       kernel_size,
                       filters,
                       stage,
                       block,
                       dilation_rate=(2, 2)):
    """A dilated conv block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = tf.keras.layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same', dilation_rate=dilation_rate,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = tf.keras.layers.Conv2D(filters3, (1, 1),
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = tf.keras.layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu', name=conv_name_base + "_add_activation")(x)
    return x



def DilatedUNet(input_size: tuple, test_mode=False) -> tf.keras.Model:
    """
    A function that builds and returns Dilated U-Net model

    Parameters
    ----------
    input_size: tuple
        The input size as tuple (width, height, n_channels)
    test_mode: bool, optional
        A status code to initiate the model for testing purpose only. (default: False)

    Returns
    -------
    tf.keras.Model
        The built Dilated U-Net model
    """

    assert len(input_size) == 3, "[ERROR]: Expected tuple of length 3 got {0}".format(len(input_size))

    image_width, image_height, n_channels = input_size

    inp = tf.keras.layers.Input(shape=(image_width, image_height, n_channels), dtype="float32", name="input_layer")

    # Get the resnet50 model
    resnet50 = tf.keras.applications.ResNet50(weights="imagenet",
                                              pooling=None,
                                              input_shape=(image_width, image_height, n_channels),
                                              include_top=False)
    resnet50_extracted = tf.keras.Model(inputs=[resnet50.inputs], outputs=[
        resnet50.get_layer("conv4_block6_out").output,
    ])
    x = __dilated_conv_block(resnet50_extracted.output, 3, [512, 512, 2048], stage=5, block='a')
    x = __dilated_identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = __dilated_identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    resnet50_dilated = tf.keras.Model(inputs=[resnet50_extracted.input], outputs=[x])
    resnet50_dilated_extracted = tf.keras.Model(inputs=[resnet50_dilated.inputs], outputs=[
        resnet50_dilated.get_layer("conv2_block3_out").output,
        resnet50_dilated.get_layer("conv3_block4_out").output,
        resnet50_dilated.get_layer("conv4_block6_out").output,
        resnet50_dilated.get_layer("res5c_branch_add_activation").output
    ])
    res2_out, res3_out, res4_out, res5_out = resnet50_dilated_extracted(inp)

    # ResNet 5 output
    res5_out = layers.Conv2DBN(256, 1, activation="relu", name="res5_out_conv")(res5_out)
    res5_out = tf.keras.layers.UpSampling2D((16, 16),
                                            interpolation="bilinear",
                                            name="res5_output")(res5_out)

    # ResNet 4 output
    res4_out = layers.Conv2DBN(48, 1, activation="relu", name="res4_out_conv")(res4_out)
    res4_out = tf.keras.layers.UpSampling2D((16, 16),
                                            interpolation="bilinear",
                                            name="res4_output")(res4_out)

    # ResNet 3 output
    res3_out = layers.Conv2DBN(48, 1, activation="relu", name="res3_out_conv")(res3_out)
    res3_out = tf.keras.layers.UpSampling2D((8, 8),
                                            interpolation="bilinear",
                                            name="res3_output")(res3_out)

    # ResNet 2 output
    res2_out = layers.Conv2DBN(48, 1, activation="relu", name="res2_out_conv")(res2_out)
    res2_out = tf.keras.layers.UpSampling2D((4, 4),
                                            interpolation="bilinear",
                                            name="res2_output")(res2_out)

    # Concatenate all output
    out = tf.keras.layers.Concatenate(axis=-1, name="output_concatenate")([res2_out,
                                                                           res3_out,
                                                                           res4_out,
                                                                           res5_out])

    out = layers.Conv2DBN(256, 3, padding="same", activation="relu", name="output_conv_1")(out)
    out = layers.Conv2DBN(128, 3, padding="same", activation="relu", name="output_conv_2")(out)
    out = layers.Conv2DBN(1, 1, padding="same", activation="sigmoid", name="output_layer")(out)

    return tf.keras.Model(inputs=[inp], outputs=[out])
