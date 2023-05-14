import numpy as np
import cv2
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard



from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

    # multiclass
    model.add(Dense(3, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.0001, momentum=0.9)

    # loss = categorical_crossentropy for multiclass
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


#early stopping callback save best model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# run the test harness for evaluating a model
def run_test_harness():
    # define model
    model = define_model()
    print(model.summary())
    # create data generators
    # create data generator using flow_from_directory

    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # prepare iterators class mode 3 for multiclass and one hot encoding label
    # train_it = train_datagen.flow_from_directory('highres_data/train/', class_mode= 'categorical', batch_size=64, target_size=(200, 200))
    # test_it = test_datagen.flow_from_directory('highres_data/test/', class_mode='categorical', batch_size=1, target_size=(200, 200))

    train_it = train_datagen.flow_from_directory('highres_data/train/', class_mode='categorical', batch_size=64, target_size=(200, 200))
    test_it = test_datagen.flow_from_directory('highres_data/test/', class_mode='categorical', batch_size=1, target_size=(200, 200))

    print(train_it.class_indices)

    #print shape of the train image and label



    # fit model
    model.fit(train_it, steps_per_epoch=len(train_it),
                                  validation_data=test_it, validation_steps=len(test_it), epochs=17, verbose=0, callbacks=[tensorboard_callback,es])
    # evaluate model
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)

    # save model as pb
    model.save("eye_classification_model_high_res5")
    print('> %.3f' % (acc * 100.0))
    # learning curves
    #summarize_diagnostics(history)

# entry point, run the test harness
run_test_harness()