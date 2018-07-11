from keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.models import Model
from keras import regularizers

input_shape = [160,160,3]

def create_enc(input_shape):
    inputs = Input(shape=input_shape)
    encoder = Conv2D(128, kernel_size=(3,3), kernel_regularizer=regularizers.l2(1e-3))(inputs)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = MaxPooling2D(pool_size=(2,2))(encoder)
    encoder = Conv2D(64, kernel_size=(3,3), kernel_regularizer=regularizers.l2(1e-3))(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = MaxPooling2D(pool_size=(2,2))(encoder)
    encoder = Conv2D(32, kernel_size=(3,3), kernel_regularizer=regularizers.l2(1e-3))(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = MaxPooling2D(pool_size=(2,2))(encoder)
    encoder = Conv2D(16, kernel_size=(3,3), kernel_regularizer=regularizers.l2(1e-3))(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = MaxPooling2D(pool_size=(2,2))(encoder)
    encoder = Conv2D(8, kernel_size=(3,3), kernel_regularizer=regularizers.l2(1e-3))(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = MaxPooling2D(pool_size=(2,2))(encoder)
    return encoder
