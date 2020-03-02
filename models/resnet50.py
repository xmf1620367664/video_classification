# _*_ coding:utf-8 _*_
from keras.applications.resnet50 import ResNet50

from keras import layers
from keras.layers import Dense
from keras import utils
from keras.models import  Model
from keras import backend as K
from attention_module import attach_attention_module

# from . import get_submodules_from_kwargs
# from . import imagenet_utils
# from .imagenet_utils import decode_predictions
# from .imagenet_utils import _obtain_input_shape

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')



def identity_block(input_tensor, kernel_size, filters, stage, block, bn_axis=3):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               bn_axis=3):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def finite_difference(input_feature):
    channel = input_feature._keras_shape[-1]
    finite_feature=layers.concatenate(
        [K.expand_dims(K.abs(layers.subtract([input_feature[...,i+1],
        input_feature[...,i]])),axis=-1) for i in range(channel-1)],axis=-1)
    return finite_feature

def res_Net50(input,classes=51,attention_module=None):
    #global backend, layers, models, keras_utils
    #backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    #x = layers.Lambda(finite_difference)(input)
    #print(x.get_shape())
    #exit()
    if attention_module is not None:
        x=attach_attention_module(input,'fcbam_block')
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input)
    x = layers.Conv2D(128, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1_he_normal')(x)
    x = layers.BatchNormalization( name='bn_conv1_he_normal')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad_he_normal')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)


    if attention_module is not None:
        x=attach_attention_module(x,attention_module)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a_he_normal', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b_he_normal')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c_he_normal')



    if attention_module is not None:
        x=attach_attention_module(x,attention_module)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')


    if attention_module is not None:
        x=attach_attention_module(x,attention_module)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')


    if attention_module is not None:
        x=attach_attention_module(x,attention_module)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    # linear = layers.Dense(units=512,activation='sigmoid',name='dense_layer_1')(x)
    # linear = layers.Dropout(rate=0.75)(linear)

    linear = layers.Dense(units=classes, activation='softmax',name='dense_layer')(x)

    model = Model(inputs=input, outputs=linear)

    weights_path = utils.get_file(
        'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        WEIGHTS_PATH_NO_TOP,
        cache_subdir='models',
        md5_hash='a268eb855778b3df3c7506639542a6af')
    model.load_weights(weights_path,by_name=True)
    return model


    #x = layers.Dense(self.classes, activation='softmax', name='fc10')(x)
    #model=Model(self.input,x,name='resnet50')
    #return model