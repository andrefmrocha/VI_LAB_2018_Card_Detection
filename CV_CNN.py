from keras import Input
import numpy as np
from keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Dropout
from keras.models import Model
from keras import optimizers, losses
from keras import regularizers
from keras import backend as K
import random

def create_base_network(input_shape, numb_conv32, numb_conv64):
    inputs = Input(shape=input_shape)
    # numb_conv32 /= 2
    # numb_conv64 /= 2
    x = Conv2D(16, kernel_size=(3,3), activation='relu' )(inputs)
    x = Dropout(0.5)(x)
    # if(numb_conv32 != 0):
    #     numb_conv32 = numb_conv32 - 1

    for i in range(numb_conv32):
        x = Conv2D(32, kernel_size=(3,3), activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
        x = Dropout(0.5)(x)
        x = Conv2D(32, kernel_size=(3,3), activation='relu',kernel_regularizer=regularizers.l2(1e-3))(x)
        x = Dropout(0.5)(x)
        # if(i%2 !=0):
        x = MaxPooling2D(pool_size=(2,2))(x)

    for i in range(numb_conv64):
        x = Conv2D(64, kernel_size=(3,3), activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
        x = Dropout(0.5)(x)
        x = Conv2D(64, kernel_size=(3,3), activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
        x = Dropout(0.5)(x)
        # if(i%2 ==0):
        x = MaxPooling2D(pool_size=(2,2))(x)
        # MaxPooling2D(pool_size=(2,2))

    opt = optimizers.Adam(lr=1e-4)

    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)

    return Model(inputs, x),opt



def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
num_classes = 2



def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

