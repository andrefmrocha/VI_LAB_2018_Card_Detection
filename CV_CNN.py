from keras import Input
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Activation, Flatten, Conv2D
from keras import optimizers, losses
from keras.regularizers import l2
import matplotlib.pyplot as plt
def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D()