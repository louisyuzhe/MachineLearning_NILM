from __future__ import print_function

from keras import layers
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import ZeroPadding1D
from keras.layers import AveragePooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import AlphaDropout
from keras.models import Model
from keras.engine.topology import get_source_inputs
import keras.backend as K


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 2

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv1D(filters1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv1D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv1D(filters3, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=2):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=2
    And the shortcut should have strides=2 as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 2

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv1D(filters1, 1, strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv1D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv1D(filters3, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv1D(filters3, 1, strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def Res50NTv1(include_top=True, weights='',
              input_tensor=None, input_shape=None,
              pooling=None, classes=1000,
              dropout=0, activation='relu',
              dense_layers=None, multi_label=False):

    if input_tensor is None:
        seq_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            seq_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            seq_input = input_tensor

    bn_axis = 2

    x = ZeroPadding1D(3)(seq_input)
    x = Conv1D(64, 7, strides=2, name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(3, strides=2)(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling1D(7, name='avg_pool', padding='same')(x)

    if include_top:
        x = Flatten()(x)
        if dense_layers is not None:
            for i, layer in enumerate(dense_layers):
                if dropout > 0 and dropout < 1:
                    dropout_layer = AlphaDropout if activation == 'selu' else Dropout
                    x = dropout_layer(dropout)(x)
                kernel_initializer = 'lecun_normal' if activation == 'selu' else 'glorot_uniform'
                name = 'fc{}'.format(i+1)
                x = Dense(layer, name=name, activation=activation, kernel_initializer=kernel_initializer)(x)
        x = Dense(input_shape[0], activation=activation, name='predictions')(x)

    else:
        if pooling == 'avg':
            x = GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = seq_input
    # Create model.
    model = Model(inputs, x, name='res50_nt')
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model


def Res50NT(include_top=True, weights='',
            input_tensor=None, input_shape=None,
            pooling=None, classes=1000,
            dropout=0, activation='relu',
            dense_layers=None, variation='v1', multi_label=False):

    fn = globals()['Res50NT' + variation]

    return fn(include_top=include_top, weights=weights,
              input_tensor=input_tensor, input_shape=input_shape,
              pooling=pooling, classes=classes,
              dropout=dropout, activation=activation,
              dense_layers=dense_layers, multi_label=multi_label)