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

def create_dec(encoder):
    decoder = Conv2DTranspose(8, kernel_size=(3, 3), activation='relu', strides=2)(encoder)
    decoder = Conv2DTranspose(16, kernel_size=(3, 3), activation='relu', strides=2)(decoder)
    decoder = Conv2DTranspose(32, kernel_size=(3, 3), activation='relu', strides=2)(decoder)
    decoder = Conv2DTranspose(64, kernel_size=(3, 3), activation='relu', strides=2)(decoder)
    decoder = Conv2DTranspose(128, kernel_size=(3, 3), activation='relu', strides=2)(decoder)
    decoder = Conv2DTranspose(64, kernel_size=(3, 3), activation='relu', strides=1)(decoder)
    decoder = Conv2DTranspose(32, kernel_size=(3, 3), activation='relu', strides=1)(decoder)
    decoder = Conv2DTranspose(16, kernel_size=(3, 3), activation='relu', strides=1)(decoder)
    decoder = Conv2DTranspose(1, kernel_size=(3, 3), activation='relu', strides=1)(decoder)
    return decoder

def create_enc_dense(input_shape):
    inputs = Input(shape=input_shape)
    encoder = Conv2D(128, kernel_size=(3,3), kernel_regularizer=regularizers.l2(1e-3),padding="same")(inputs)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = MaxPooling2D(pool_size=(2,2))(encoder)
    encoder = Conv2D(64, kernel_size=(3,3), kernel_regularizer=regularizers.l2(1e-3),padding="same")(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = MaxPooling2D(pool_size=(2,2))(encoder)
    encoder = Conv2D(32, kernel_size=(3,3), kernel_regularizer=regularizers.l2(1e-3),padding="same")(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = MaxPooling2D(pool_size=(2,2))(encoder)
    encoder = Conv2D(16, kernel_size=(3,3), kernel_regularizer=regularizers.l2(1e-3),padding="same")(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = MaxPooling2D(pool_size=(2,2))(encoder)
    encoder = Conv2D(8, kernel_size=(3,3), kernel_regularizer=regularizers.l2(1e-3),padding="same")(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = MaxPooling2D(pool_size=(2,2))(encoder)
    encoder = Flatten()(encoder)
    encoder = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(encoder)
    return Model(inputs=inputs, outputs=encoder)
#(6,6,8)
def create_dec_dense(input_shape=(256,), img_size=(5,5,8)):
    inputs = Input(shape=input_shape)
    decoder = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(inputs)
    decoder = Dense(200, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(decoder)
    decoder = Reshape(img_size)(decoder)
    decoder = Conv2DTranspose(8, kernel_size=(3,3), activation='relu', strides=2, padding="same")(decoder)
    decoder = Conv2DTranspose(16, kernel_size=(3,3), activation='relu', strides=2, padding="same")(decoder)
    decoder = Conv2DTranspose(32, kernel_size=(3,3), activation='relu', strides=2, padding="same")(decoder)
    decoder = Conv2DTranspose(64, kernel_size=(3,3), activation='relu', strides=2, padding="same")(decoder)
    decoder = Conv2DTranspose(128, kernel_size=(3,3), activation='relu', strides=2, padding="same")(decoder)
    decoder = Conv2DTranspose(64, kernel_size=(3,3), activation='relu', strides=1, padding="same")(decoder)
    decoder = Conv2DTranspose(32, kernel_size=(3,3), activation='relu', strides=1, padding="same")(decoder)
    decoder = Conv2DTranspose(16, kernel_size=(3,3), activation='relu', strides=1, padding="same")(decoder)
    decoder = Conv2DTranspose(3, kernel_size=(3,3), activation='relu', strides=1, padding="same")(decoder)
    return Model(inputs=inputs, outputs=decoder)
