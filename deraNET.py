from keras.layers import Input,Activation, Add , Dense, Convolution2D, MaxPooling2D, UpSampling2D, Deconv2D, \
    Flatten, Dropout, SpatialDropout2D, Reshape
from keras.models import Model, Sequential
from keras import backend as K
from keras.activations import tanh
from keras.optimizers import adam
from keras.regularizers import l2

opt = adam(lr=1e-5)
reg = l2(3e-4)


def new_autoenc():
    input_img = Input(shape=(256,256,3))

    x = Convolution2D(32, kernel_size=(3, 3), activation='elu', padding='valid') (input_img)
    x = Convolution2D(32, kernel_size=(4, 4), strides=(2,2), activation='elu', padding='valid')(x)
    x = SpatialDropout2D(0.3) (x)
    x = Convolution2D(32, kernel_size=(3, 3), activation='elu', padding='valid')(x)
    x = Convolution2D(32, kernel_size=(3, 3), strides=(2, 2), activation='elu', padding='same')(x)
    x = SpatialDropout2D(0.3)(x)
    encoded = Convolution2D(64, kernel_size=(4, 4), strides=(2,2), activation='elu', padding='same')(x)

    print("shape of encoded", K.int_shape(encoded))

    x = Deconv2D(64, kernel_size=(4, 4), strides=(2,2), activation='elu', padding='same')(encoded)
    x = SpatialDropout2D(0.3)(x)
    x = Deconv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='elu', padding='same')(x)
    x = Deconv2D(32, kernel_size=(3, 3), activation='elu', padding='valid')(x)
    x = SpatialDropout2D(0.3)(x)
    x = Deconv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='elu', padding='valid')(x)
    decoded = Deconv2D(3, kernel_size=(3, 3), padding='valid')(x)
    # output = Add() ([decoded, input_img])
    # act = Activation('tanh') (output)

    print("shape of decoded", K.int_shape(decoded))

    ae = Model(input_img, decoded)
    # ae.compile(optimizer=opt, loss='mse')
    return ae

def generator():
    inp_noise = Input(shape=(100,))
    upsize = Dense(4*4*512, activation='elu',
                            kernel_regularizer=reg) (inp_noise)
    reshape2d = Reshape((4,4,512)) (upsize)
    deconv = Deconv2D(384, kernel_size=(5, 5), strides=(2,2), activation='elu', padding='same',
                            kernel_regularizer=reg)(reshape2d)
    deconv = Deconv2D(256, kernel_size=(5, 5), strides=(2, 2), activation='elu', padding='same',
                            kernel_regularizer=reg)(deconv)
    deconv = Deconv2D(128, kernel_size=(5, 5), strides=(2, 2), activation='elu', padding='same',
                            kernel_regularizer=reg)(deconv)
    deconv = Deconv2D(96, kernel_size=(5, 5), strides=(2, 2), activation='elu', padding='same',
                            kernel_regularizer=reg)(deconv)
    deconv = Deconv2D(64, kernel_size=(5, 5), strides=(2, 2), activation='elu', padding='same',
                            kernel_regularizer=reg)(deconv)
    img = Deconv2D(3, kernel_size=(5, 5), strides=(2, 2), activation='sigmoid', padding='same',
                            kernel_regularizer=reg)(deconv)

    g = Model(inp_noise, img)

    return g





def new_discriminator():
    model = Sequential()
    model.add(Convolution2D(64, kernel_size=(3, 3),input_shape=(256,256,3), activation='elu',
                            kernel_regularizer=reg))
    model.add(Convolution2D(128, kernel_size=(4, 4), strides=(2,2), activation='elu',
                            kernel_regularizer=reg))
    model.add(Convolution2D(256, kernel_size=(3, 3), strides=(2,2),activation='elu',
                            kernel_regularizer=reg))
    model.add(Convolution2D(256, kernel_size=(4, 4), strides=(2,2), activation='elu',
                            kernel_regularizer=reg))
    model.add(Convolution2D(512, kernel_size=(4, 4), strides=(2,2), activation='elu',
                            kernel_regularizer=reg))
    model.add(Convolution2D(256, kernel_size=(3, 3), strides=(2, 2), activation='elu',
                            kernel_regularizer=reg))
    model.add(Flatten())
    model.add(Dense(256, activation='elu',
                            kernel_regularizer=reg))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt, loss='binary_crossentropy')
    return model

def gan(ae, discr):
    model = Sequential()
    model.add(ae)
    discr.trainable = False
    model.add(discr)
    model.compile(optimizer=opt, loss= 'binary_crossentropy')
    return model

b = new_discriminator()
b.summary()