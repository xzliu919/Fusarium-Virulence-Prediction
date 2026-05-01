# code/utils/ResGS.py
# 纯净版：移除了所有与新流水线冲突的外部依赖

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings('ignore')

def Conv1d_BN(x, nb_filter, kernel_size, strides=1):
    x = layers.Convolution1D(nb_filter, kernel_size, padding='same', strides=strides, activation='relu')(x)
    x = layers.BatchNormalization(axis=1)(x)
    return x

def Res_Block(inpt, nb_filter, kernel_size, strides=1):
    x = Conv1d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides)
    x = layers.add([x, inpt])
    return x

def ResGSModel(inputs):
    nFilter = 64
    _KERNEL_SIZE = 3
    CHANNEL_FACTOR1 = 4
    CHANNEL_FACTOR2 = 1.1

    x1 = Res_Block(inputs, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x1 = Res_Block(x1, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    nFilter1 = int(nFilter * CHANNEL_FACTOR1)

    x2 = Conv1d_BN(x1 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x2 = Conv1d_BN(x2, nb_filter=nFilter, kernel_size=1, strides=1)
    x2 = Res_Block(x2, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x2 = Res_Block(x2, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)

    x3 = Conv1d_BN(x2 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x3 = Conv1d_BN(x3, nb_filter=nFilter, kernel_size=1, strides=1)
    x3 = Res_Block(x3, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x3 = Res_Block(x3, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)

    x4 = Conv1d_BN(x3 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x4 = Conv1d_BN(x4, nb_filter=nFilter, kernel_size=1, strides=1)
    x4 = Res_Block(x4, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x4 = Res_Block(x4, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)

    x5 = Conv1d_BN(x4 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x5 = Conv1d_BN(x5, nb_filter=nFilter, kernel_size=1, strides=1)
    x5 = Res_Block(x5, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x5 = Res_Block(x5, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)

    x6 = Conv1d_BN(x5 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x6 = Conv1d_BN(x6, nb_filter=nFilter, kernel_size=1, strides=1)
    x6 = Res_Block(x6, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x6 = Res_Block(x6, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)

    x7 = Conv1d_BN(x6 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x7 = Conv1d_BN(x7, nb_filter=nFilter, kernel_size=1, strides=1)
    x7 = Res_Block(x7, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x7 = Res_Block(x7, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)

    x8 = Conv1d_BN(x7 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x8 = Conv1d_BN(x8, nb_filter=nFilter, kernel_size=1, strides=1)
    x8 = Res_Block(x8, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x8 = Res_Block(x8, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)

    x9 = Conv1d_BN(x8 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x9 = Conv1d_BN(x9, nb_filter=nFilter, kernel_size=1, strides=1)
    x9 = Res_Block(x9, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x9 = Res_Block(x9, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)

    element_number = x9.shape[1] * x9.shape[2]
    filter_near_6400 = 6400 // x9.shape[1]
    if filter_near_6400 == 0:
        filter_near_6400 = 1

    x = Conv1d_BN(x9, nb_filter=filter_near_6400, kernel_size=1, strides=1)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    return Model(inputs=inputs, outputs=x)