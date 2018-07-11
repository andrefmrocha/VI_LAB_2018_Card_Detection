from keras.layers import Input, Conv2D, MaxPooling2D, \
    Activation, BatchNormalization, Conv2DTranspose, \
    Flatten, Dense, Reshape

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

def create_enc_dense(input_shape):
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
    encoder = Flatten()(encoder)
    encoder = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(encoder)
    return encoder
#(6,6,8)
def create_dec(encoder):
    decoder = Conv2DTranspose(8, kernel_size=(3,3), activation='relu', strides=2)(encoder)
    decoder = Conv2DTranspose(16, kernel_size=(3,3), activation='relu', strides=2)(decoder)
    decoder = Conv2DTranspose(32, kernel_size=(3,3), activation='relu', strides=2)(decoder)
    decoder = Conv2DTranspose(64, kernel_size=(3,3), activation='relu', strides=2)(decoder)
    decoder = Conv2DTranspose(128, kernel_size=(3,3), activation='relu', strides=2)(decoder)
    decoder = Reshape((160,160,3))(decoder)
    return decoder

def create_dec_dense(input_shape):
    decoder = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-3), input_shape=(256,))
    decoder = Reshape((6,6,8))(decoder)
    decoder = Conv2DTranspose(8, kernel_size=(3,3), activation='relu', strides=2)(decoder)
    decoder = Conv2DTranspose(16, kernel_size=(3,3), activation='relu', strides=2)(decoder)
    decoder = Conv2DTranspose(32, kernel_size=(3,3), activation='relu', strides=2)(decoder)
    decoder = Conv2DTranspose(64, kernel_size=(3,3), activation='relu', strides=2)(decoder)
    decoder = Conv2DTranspose(128, kernel_size=(3,3), activation='relu', strides=2)(decoder)
    return decoder
