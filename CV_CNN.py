from keras import Input
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Activation, Flatten, Conv2D
from keras import optimizers, losses
from keras.regularizers import l2
import matplotlib.pyplot as plt

def create_base_network(input_shape, numb_conv32, numb_conv64):
    inputs = Input(shape=input_shape)

    x = Conv2(16, kernel_size={3,3}, activation='relu' )(inputs)

    
    for i in range(numb_conv32):
        x= Conv2(32, kernel_size={3,3}, activation='relu')(x)
        if(i%2 ==0):
            MaxPooling2D(pool_size=(2,2)) 
    
    for i in range(numb_conv64):
        x= Conv2(64, kernel_size={3,3}, activation='relu')(x)
        if(i%2 ==0):
            MaxPooling2D(pool_size=(2,2)) 
        
    return Model(inputs, x)

