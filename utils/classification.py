from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, SpatialDropout2D
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np

def model(img_ch, img_width, img_height, n_base):
    
    model = Sequential()
    
    model.add(Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=n_base*2, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=n_base*4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=n_base*4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=n_base*2, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.summary()   
    return model

def train_classifier(train_img, train_labels, val_img, val_labels, learning_rate, n_epochs): 
    """
    Receive a list of images `im_list` and a list of vectors (one per image) 
    with the labels 0 or 1 depending on the sagittal 2D slice contains or not 
    the obturator foramen. Returns the trained classifier.
    """ 

    batch_size = 8
    n_base = 8
    MLP = model(1, 256, 512, n_base)
    MLP.compile(loss = 'categorical_crossentropy',         
              optimizer = Adam(lr = learning_rate),
              metrics = ['accuracy'])
    
    history = MLP.fit(train_img, train_labels, 
                      batch_size = batch_size,
                      validation_data = (val_img, val_labels),
                      epochs = n_epochs, verbose=1)
    
    return MLP, history

