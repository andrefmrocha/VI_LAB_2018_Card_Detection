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
x_Train_Card, x_Test_Card, y_Train, y_Test= train_test_split(list_card, list_labels, test_size=0.4, random_state=1)
x_Test_Card, x_Val_Card, y_Test, y_Val = train_test_split(x_Test_Card,y_Test, test_size=0.5, random_state=1)
x_Train_Camera, x_Test_Camera= train_test_split(list_camera, test_size=0.4, random_state=1)
x_Test_Camera, x_Val_Camera= train_test_split(x_Test_Camera, test_size=0.5, random_state=1)

x_Train = {"card": x_Train_Card, "camera": x_Train_Camera}
x_Test = {"card": x_Test_Card, "camera": x_Test_Camera}
x_Val = {"card": x_Val_Card, "camera": x_Val_Camera}