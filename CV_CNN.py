from CV_Dataset import *
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Activation, Flatten, Conv2D
from keras import optimizers, losses
from keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
x_Card, x_Camera = open_images()
list_card, list_camera, list_labels = gen_data(x_Card, x_Camera)
# 60% - train, 20%, 20% - test, validation