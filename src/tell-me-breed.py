import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import keras
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Flatten, Dense, BatchNormalization, Dropout, Lambda, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import xception, inception_v3, vgg16, resnet50
from sklearn.linear_model import LogisticRegression
import cv2

from keras.datasets import mnist
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

image_size = 224
input_shape = (image_size, image_size, 3)
batch_size = 24
classes_count = 120
model_name = "vgg16-fc1-fc2"
input_dir = "../input"
output_dir = "../output"
augmented_dir = output_dir + "/augmented"
augmented_dir = None

train_datagen = ImageDataGenerator(preprocessing_function=keras.applications.vgg19.preprocess_input,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   rotation_range=25,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2)
train_generator = train_datagen.flow_from_directory(input_dir + "/sorted/train",
                                                    shuffle=False,
                                                    target_size=(image_size, image_size),
                                                    batch_size=batch_size,
                                                    save_to_dir=augmented_dir,
                                                    seed=1)

val_datagen = ImageDataGenerator(preprocessing_function=keras.applications.vgg19.preprocess_input)
validation_generator = val_datagen.flow_from_directory(input_dir + "/sorted/validate",
                                                       shuffle=False,
                                                       target_size=(image_size, image_size),
                                                       batch_size=batch_size,
                                                       seed=1)


def build_base_model(input_shape, dropout=0.8):
    full_vgg16 = vgg16.VGG16()

    fc1_pretrained = full_vgg16.get_layer("fc1")
    fc1_pretrained.trainable = False
    fc2_pretrained = full_vgg16.get_layer("fc2")
    fc2_pretrained.trainable = False

    m = vgg16.VGG16(include_top=False, input_shape=input_shape)
    for layer in m.layers:
        layer.trainable = False

    x = Flatten(name="tail_flatten")(m.output)
    x = fc1_pretrained(x)
    # x = fc2_pretrained(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    return Model(m.input, x)


def add_model_tail(base_model, output_classes_count, name, dropout=0.8):
    x = Dense(512, activation='relu', name="tail_fc1")(base_model.output)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    # x = Dense(512, activation='relu', name='tail_fc2')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(dropout)(x)
    #
    # x = Dense(256, activation='relu', name='tail_fc3')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(dropout)(x)

    x = Dense(output_classes_count, name="tail_fc_final", activation='softmax')(x)
    return Model(base_model.input, x, name=name)


def load_trained_model(input_shape, output_classes_count, name):
    model = add_model_tail(base_model=build_base_model(input_shape=input_shape),
                           output_classes_count=output_classes_count,
                           name=name)
    model.load_weights("{}/weights/{}-weights.hdf5".format(output_dir, model_name))
    return model


def predict(trained_model):
    datagen = ImageDataGenerator(preprocessing_function=keras.applications.vgg19.preprocess_input)
    generator = datagen.flow_from_directory(input_dir + "/test",
                                            shuffle=False,
                                            target_size=(image_size, image_size))
    return trained_model.predict_generator(generator,
                                           verbose=1)


def create_submission(predictions, input_folder, output_folder, model_name):
    labels_csv = pd.read_csv(input_folder + "/labels.csv")
    breeds = pd.Series(labels_csv['breed'])
    unique_breeds = np.unique(breeds)
    unique_breeds.sort()

    class_to_num = dict(zip(unique_breeds, range(unique_breeds.size)))
    df2 = pd.read_csv(input_folder + "/sample_submission.csv")

    for b in unique_breeds:
        df2[b] = predictions[:, class_to_num[b]]
    df2.to_csv("{}/submissions/{}-submission.csv".format(output_folder, model_name), index=None)
    exit(0)


create_submission(predict(load_trained_model(input_shape=input_shape,
                                             output_classes_count=classes_count,
                                             name=model_name)),
                  input_dir,
                  output_dir,
                  model_name)
exit(0)

model = add_model_tail(base_model=build_base_model(input_shape=input_shape),
                       output_classes_count=classes_count,
                       name=model_name)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
weight_saving = ModelCheckpoint(filepath="{}/weights/{}-weights.hdf5".format(output_dir, model_name),
                                verbose=1,
                                save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=2,
                              min_lr=0.001)
callbacks = [early_stopping, weight_saving, reduce_lr]
# callbacks = None

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=int(len(train_generator) / batch_size),
                              epochs=50,
                              validation_data=validation_generator,
                              validation_steps=int(len(validation_generator) / batch_size),
                              shuffle=False,
                              verbose=2,
                              callbacks=callbacks)


# fig, ax = plt.subplots(2, 1)
# ax[0].plot(history.history['loss'], color='b', label="Training loss")
# ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
# legend = ax[0].legend(loc='best', shadow=True)
#
# ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
# ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
# legend = ax[1].legend(loc='best', shadow=True)
