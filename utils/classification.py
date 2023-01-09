from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, SpatialDropout2D
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np
import matplotlib.pyplot as plt

def AlexNet(img_ch, img_width, img_height, n_base=8, dropout=True, batch_norm=True):

    """
    This function is used to resample the input image mask with the given size.

    Parameters
    ----------
    img_ch : 
        The number of image chanels
    img_width :
        Image width
    img_height :
        Image height
    n_base :
        Number of neurons in the first convolution layer
    dropout : True or False
        Include dropout layers
    batch_norm : True or False
        Include batch normalization layers

    Returns
    -----------
    The model ready for the training.

    """

    model = Sequential()

    model.add(Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch),
    kernel_size=(3,3), strides=(1,1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
   
    model.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same', name = 'Last_ConvLayer'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(128))
    if dropout:
        model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(64))
    if dropout:
        model.add(Dropout(0.4))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()

    model.compile(loss = 'binary_crossentropy',         
            optimizer = Adam(lr = 1e-5),
            metrics = ['accuracy'])

    return model

def load_train_data(imgs):
    # imgs should be in lists of 53,54,55
    x_train = []
    y_train = []
    for img in imgs:
        for idx in range(512):
            slice = img[:,:,idx]
            x_train.append(slice)
    x_train = np.array(x_train)

    # Generate ground truth for training data
    y_53 = np.zeros(512)
    y_53[173:212] = 1
    y_53[312:345] = 1

    y_54 = np.zeros(512)
    y_54[160:203] = 1
    y_54[290:326] = 1

    y_55 = np.zeros(512)
    y_55[181:225] = 1
    y_55[292:331] = 1

    y_train = np.concatenate((y_53, y_54, y_55),axis=0)

    return x_train, y_train
    
def load_test_data(img):
    x_test = []
    for idx in range(512):
        slice = img[:,:,idx]
        x_test.append(slice)
    x_test = np.array(x_test)
    return x_test

def plot_history(history):
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(history.history["val_loss"]),
            np.min(history.history["val_loss"]),
            marker="x", color="r", label="best model")

    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.show()

    plt.figure(figsize=(4, 4))
    plt.title("Accuracy")
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.plot( np.argmax(history.history["val_accuracy"]),
            np.max(history.history["val_accuracy"]),
            marker="x", color="r", label="best model")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

