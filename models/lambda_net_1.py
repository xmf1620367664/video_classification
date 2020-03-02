# _*_ coding:utf-8 _*_
import numpy as np

from keras import layers
from keras.layers import SeparableConv2D as Conv2D
from keras import backend as K
from keras.models import  Model
from .attention_module import attach_attention_module
from keras.applications import xception

np.random.seed(36)

def Bottleneck_25(input_tensor,filters=16,attention_module='cbam_block'):
    assert input_tensor.get_shape()[-1]==1
    #经过zeropadding,允许每一个通道的第一个卷积核，padding='valid'
    x1=layers.Conv2D(filters,(1,1),kernel_initializer='he_normal',padding='same')(input_tensor)
    x1=layers.BatchNormalization()(x1)
    x1=layers.Activation('relu')(x1)

    x2 = layers.Conv2D(filters,(1,1),kernel_initializer='he_normal')(input_tensor)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = Conv2D(filters,(3,3),kernel_initializer='he_normal',padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)

    x3 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal')(input_tensor)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)
    x3 = Conv2D(filters, (5, 5), kernel_initializer='he_normal', padding='same')(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)

    x4 = layers.MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(input_tensor)
    x4 = layers.Conv2D(filters,(1,1),kernel_initializer='he_normal',padding='same')(x4)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Activation('relu')(x4)

    x = layers.concatenate([x1,x2,x3,x4],axis=-1)
    if attention_module is not None:
        x=attach_attention_module(x,attention_module=attention_module)
    return x

def lambda_25(input_tensor,attention_module='cbam_block'):
    channel=input_tensor._keras_shape[-1]
    maps=[]
    for i in range(channel):
        curr_feature=K.expand_dims(input_tensor[...,i],axis=-1)
        maps.append(Bottleneck_25(curr_feature))
    features=layers.concatenate(maps,axis=-1)
    if attention_module is not None:
        features=attach_attention_module(features,attention_module)

    #shuffle
    assert features._keras_shape[-1]==64*25
    random_list=np.random.permutation(64*25)
    shuffle_features=[]
    for i in random_list:
        shuffle_features.append(K.expand_dims(features[...,i],axis=-1))
    random_features=layers.concatenate(shuffle_features,axis=-1)
    return random_features

def Bottleneck_16_8_4_2(input_tensor,filters=50,attention_module='cbam_block'):
    assert input_tensor.get_shape()[-1] == 100
    # 经过zeropadding,允许每一个通道的第一个卷积核，padding='valid'
    x1 = layers.Conv2D(filters//5, (1, 1), kernel_initializer='he_normal',padding='same')(input_tensor)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)

    x2 = layers.Conv2D(2*(filters//5), (1, 1), kernel_initializer='he_normal',padding='same')(input_tensor)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = Conv2D(2*(filters//5), (3, 3), kernel_initializer='he_normal', padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)

    x3 = layers.Conv2D(filters//5, (1, 1), kernel_initializer='he_normal',padding='same')(input_tensor)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)
    x3 = Conv2D(filters//5, (5, 5), kernel_initializer='he_normal', padding='same')(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)

    x4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1),padding='same')(input_tensor)
    x4 = layers.Conv2D(filters//5, (1, 1), kernel_initializer='he_normal', padding='same')(x4)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Activation('relu')(x4)

    x = layers.concatenate([x1, x2, x3, x4], axis=-1)
    if attention_module is not None:
        x = attach_attention_module(x, attention_module=attention_module)
    return x


def lambda_16(input_tensor,attention_module='cbam_block'):
    assert input_tensor._keras_shape[-1]==100*16
    size=64*25//16
    assert size==100
    maps=[]
    for i in range(16):
        curr_features=input_tensor[...,i*size:(i+1)*size]
        maps.append(Bottleneck_16_8_4_2(curr_features))
    features=layers.concatenate(maps,axis=-1)
    if attention_module is not None:
        features=attach_attention_module(features,attention_module)

    # shuffle
    assert features.get_shape()[-1] == 50 * 16
    random_list = np.random.permutation(50 * 16)
    shuffle_features = []
    for i in random_list:
        shuffle_features.append(K.expand_dims(features[..., i], axis=-1))
    random_features = layers.concatenate(shuffle_features, axis=-1)
    return random_features

def lambda_8(input_tensor,attention_module='cbam_block'):
    assert input_tensor.get_shape()[-1]==100*8
    size=100*8//8
    assert size==100
    maps=[]
    for i in range(8):
        curr_features = input_tensor[..., i * size:(i + 1) * size]
        maps.append(Bottleneck_16_8_4_2(curr_features))
    features = layers.concatenate(maps, axis=-1)
    if attention_module is not None:
        features = attach_attention_module(features, attention_module)

    # shuffle
    assert features.get_shape()[-1] == 50 * 8
    random_list = np.random.permutation(50 * 8)
    shuffle_features = []
    for i in random_list:
        shuffle_features.append(K.expand_dims(features[..., i], axis=-1))
    random_features = layers.concatenate(shuffle_features, axis=-1)
    return random_features

def lambda_4(input_tensor,attention_module='cbam_block'):
    assert input_tensor.get_shape()[-1] == 100 * 4
    size = 100 * 4 // 4
    assert size == 100
    maps = []
    for i in range(4):
        curr_features = input_tensor[..., i * size:(i + 1) * size]
        maps.append(Bottleneck_16_8_4_2(curr_features))
    features = layers.concatenate(maps, axis=-1)
    if attention_module is not None:
        features = attach_attention_module(features, attention_module)

    # shuffle
    assert features.get_shape()[-1] == 50 * 4
    random_list = np.random.permutation(50 * 4)
    shuffle_features = []
    for i in random_list:
        shuffle_features.append(K.expand_dims(features[..., i], axis=-1))
    random_features = layers.concatenate(shuffle_features, axis=-1)
    return random_features

def lambda_2(input_tensor,attention_module='cbam_block'):
    assert input_tensor.get_shape()[-1] == 100 * 2
    size = 100 * 2 // 2
    assert size == 100
    maps = []
    for i in range(2):
        curr_features = input_tensor[..., i * size:(i + 1) * size]
        maps.append(Bottleneck_16_8_4_2(curr_features))
    features = layers.concatenate(maps, axis=-1)
    if attention_module is not None:
        features = attach_attention_module(features, attention_module)

    # shuffle
    assert features.get_shape()[-1] == 50 * 2
    random_list = np.random.permutation(50 * 2)
    shuffle_features = []
    for i in random_list:
        shuffle_features.append(K.expand_dims(features[..., i], axis=-1))
    random_features = layers.concatenate(shuffle_features, axis=-1)
    return random_features


def lambda_net(input_tensor,classes=51,attention_module='cbam_block'):

    #x=layers.BatchNormalization()(input)

    x=layers.AveragePooling2D(pool_size=(3,3),strides=(2,2))(input_tensor)

    #x=layers.ZeroPadding2D((1,1))(input)
    x=layers.Lambda(lambda_25)(x)

    x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    # attention ??? 是否需要添加   1600
    x=layers.Lambda(lambda_16)(x)
    x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    #800
    x=layers.Lambda(lambda_8)(x)
    x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    #400
    L1=layers.GlobalAveragePooling2D()(x)

    x=layers.Lambda(lambda_4)(x)
    x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    #200
    L2=layers.GlobalAveragePooling2D()(x)

    x=layers.Lambda(lambda_2)(x)
    x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    #100
    L3=layers.GlobalAveragePooling2D()(x)

    linear=layers.concatenate([L1,L2,L3],axis=-1)

    linear = layers.Dense(units=classes, activation='softmax', name='fc_layer')(linear)

    model = Model(inputs=input_tensor, outputs=linear)
    return model