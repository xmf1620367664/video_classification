# _*_ coding:utf-8 _*_
import numpy as np

from keras import layers
from keras.layers import SeparableConv2D as Conv2D
from keras import backend as K
from keras.models import  Model
from .attention_module import attach_attention_module

np.random.seed(36)

def Bottleneck_25(input_tensor,filters=2,attention_module='cbam_block'):
    assert input_tensor.get_shape()[-1]==1
    x1 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal', padding='same')(input_tensor)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)

    x2 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal')(input_tensor)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)

    x3 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal')(input_tensor)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)
    x3 = Conv2D(filters, (5, 5), kernel_initializer='he_normal', padding='same')(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)

    x4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    x4 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal', padding='same')(x4)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Activation('relu')(x4)

    x = layers.concatenate([x1, x2, x3, x4], axis=-1)
    if attention_module is not None:
        x = attach_attention_module(x, attention_module=attention_module)
    return x

def Bottleneck_200(input_tensor,filters=20,attention_module='cbam_block'):
    assert input_tensor.get_shape()[-1] == 40
    x1 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal', padding='same')(input_tensor)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)

    x2 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal')(input_tensor)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)

    x3 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal')(input_tensor)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)
    x3 = Conv2D(filters, (5, 5), kernel_initializer='he_normal', padding='same')(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)

    x4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    x4 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal', padding='same')(x4)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Activation('relu')(x4)

    x = layers.concatenate([x1, x2, x3, x4], axis=-1)
    if attention_module is not None:
        x = attach_attention_module(x, attention_module=attention_module)
    return x

def Bottleneck_400(input_tensor,filters=40,attention_module='cbam_block'):
    assert input_tensor.get_shape()[-1] == 80
    x1 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal', padding='same')(input_tensor)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)

    x2 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal')(input_tensor)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)

    x3 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal')(input_tensor)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)
    x3 = Conv2D(filters, (5, 5), kernel_initializer='he_normal', padding='same')(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)

    x4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    x4 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal', padding='same')(x4)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Activation('relu')(x4)

    x = layers.concatenate([x1, x2, x3, x4], axis=-1)
    if attention_module is not None:
        x = attach_attention_module(x, attention_module=attention_module)
    return x

def Bottleneck_800(input_tensor,filters=20,attention_module='cbam_block'):
    assert input_tensor.get_shape()[-1] == 80
    x1 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal', padding='same')(input_tensor)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)

    x2 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal')(input_tensor)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)

    x3 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal')(input_tensor)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)
    x3 = Conv2D(filters, (5, 5), kernel_initializer='he_normal', padding='same')(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)

    x4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    x4 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal', padding='same')(x4)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Activation('relu')(x4)

    x = layers.concatenate([x1, x2, x3, x4], axis=-1)
    if attention_module is not None:
        x = attach_attention_module(x, attention_module=attention_module)
    return x

def Bottleneck_1600(input_tensor,filters=40,attention_module='cbam_block'):
    assert input_tensor.get_shape()[-1] == 160
    x1 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal', padding='same')(input_tensor)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)

    x2 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal')(input_tensor)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)

    x3 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal')(input_tensor)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)
    x3 = Conv2D(filters, (5, 5), kernel_initializer='he_normal', padding='same')(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)

    x4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    x4 = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal', padding='same')(x4)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Activation('relu')(x4)

    x = layers.concatenate([x1, x2, x3, x4], axis=-1)
    if attention_module is not None:
        x = attach_attention_module(x, attention_module=attention_module)
    return x

def lambda_25(input_tensor,attention_module='cbam_block'):
    channel = input_tensor._keras_shape[-1]
    maps = []
    for i in range(channel):
        curr_feature = K.expand_dims(input_tensor[..., i], axis=-1)
        maps.append(Bottleneck_25(curr_feature))
    features = layers.concatenate(maps, axis=-1)
    if attention_module is not None:
        features = attach_attention_module(features, attention_module)

    # shuffle
    assert features._keras_shape[-1] == 8 * 25
    random_list = np.random.permutation(8 * 25)
    shuffle_features = []
    for i in random_list:
        shuffle_features.append(K.expand_dims(features[..., i], axis=-1))
    random_features = layers.concatenate(shuffle_features, axis=-1)
    return random_features

def lambda_200(input_tensor,attention_module='cbam_block',group=5):
    assert input_tensor._keras_shape[-1]==200
    size=200//group
    maps = []
    for i in range(group):
        curr_features = input_tensor[..., i * size:(i + 1) * size]
        maps.append(Bottleneck_200(curr_features))
    features = layers.concatenate(maps, axis=-1)
    if attention_module is not None:
        features = attach_attention_module(features, attention_module)

    # shuffle
    assert features.get_shape()[-1] == 80 * 5
    random_list = np.random.permutation(80 * 5)
    shuffle_features = []
    for i in random_list:
        shuffle_features.append(K.expand_dims(features[..., i], axis=-1))
    random_features = layers.concatenate(shuffle_features, axis=-1)
    return random_features

def lambda_400(input_tensor,attention_module='cbam_block',group=5):
    assert input_tensor._keras_shape[-1] == 400
    size = 400 // group
    maps = []
    for i in range(group):
        curr_features = input_tensor[..., i * size:(i + 1) * size]
        maps.append(Bottleneck_400(curr_features))
    features = layers.concatenate(maps, axis=-1)
    if attention_module is not None:
        features = attach_attention_module(features, attention_module)

    # shuffle
    assert features.get_shape()[-1] == 160 * 5
    random_list = np.random.permutation(160 * 5)
    shuffle_features = []
    for i in random_list:
        shuffle_features.append(K.expand_dims(features[..., i], axis=-1))
    random_features = layers.concatenate(shuffle_features, axis=-1)
    return random_features

def lambda_800(input_tensor,attention_module='cbam_block',group=10):
    assert input_tensor._keras_shape[-1] == 800
    size = 800 // group
    maps = []
    for i in range(group):
        curr_features = input_tensor[..., i * size:(i + 1) * size]
        maps.append(Bottleneck_800(curr_features))
    features = layers.concatenate(maps, axis=-1)
    if attention_module is not None:
        features = attach_attention_module(features, attention_module)

    # shuffle
    assert features.get_shape()[-1] == 80 * 10
    random_list = np.random.permutation(80 * 10)
    shuffle_features = []
    for i in random_list:
        shuffle_features.append(K.expand_dims(features[..., i], axis=-1))
    random_features = layers.concatenate(shuffle_features, axis=-1)
    return random_features

def lambda_800_tail(input_tensor,attention_module='cbam_block',group=10):
    assert input_tensor._keras_shape[-1] == 800
    size = 800 // group
    maps = []
    for i in range(group):
        curr_features = input_tensor[..., i * size:(i + 1) * size]
        maps.append(Bottleneck_800(curr_features,filters=40))
    features = layers.concatenate(maps, axis=-1)
    if attention_module is not None:
        features = attach_attention_module(features, attention_module)

    # shuffle
    assert features.get_shape()[-1] == 160 * 10
    random_list = np.random.permutation(160 * 10)
    shuffle_features = []
    for i in random_list:
        shuffle_features.append(K.expand_dims(features[..., i], axis=-1))
    random_features = layers.concatenate(shuffle_features, axis=-1)
    return random_features

def lambda_1600(input_tensor,attention_module='cbam_block',group=10):
    assert input_tensor._keras_shape[-1] == 1600
    size = 1600 // group
    maps = []
    for i in range(group):
        curr_features = input_tensor[..., i * size:(i + 1) * size]
        maps.append(Bottleneck_1600(curr_features, filters=40))
    features = layers.concatenate(maps, axis=-1)
    if attention_module is not None:
        features = attach_attention_module(features, attention_module)

    # shuffle
    assert features.get_shape()[-1] == 160 * 10
    random_list = np.random.permutation(160 * 10)
    shuffle_features = []
    for i in random_list:
        shuffle_features.append(K.expand_dims(features[..., i], axis=-1))
    random_features = layers.concatenate(shuffle_features, axis=-1)
    return random_features

def lambda_net(input_tensor,classes=51,attention_module='cbam_block'):
    x=layers.Lambda(lambda_25)(input_tensor)
    x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)

    x=layers.Lambda(lambda_200)(x)
    x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)

    x=layers.Lambda(lambda_400)(x)
    x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)

    x=layers.Lambda(lambda_800)(x)
    x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)

    for i in range(4):
        x=layers.Lambda(lambda_800)(x)

    x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)
    x=layers.Lambda(lambda_800_tail)(x)

    x=layers.Lambda(lambda_1600)(x)
    x=layers.GlobalAveragePooling2D()(x)

    linear = layers.Dense(units=classes, activation='softmax', name='fc_layer',kernel_initializer='he_normal')(x)

    model = Model(inputs=input_tensor, outputs=linear)
    return model
