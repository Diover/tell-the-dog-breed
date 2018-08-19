import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import keras
from keras import Input, Model
from keras.applications.xception import decode_predictions
from keras.backend import resize_images
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Flatten, Dense, BatchNormalization, Dropout, Lambda, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Average, Reshape, Cropping2D, Conv2D, Activation, MaxPooling2D, Maximum, Concatenate
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import xception, inception_v3, vgg16, resnet50
from sklearn.linear_model import LogisticRegression
import cv2
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from keras.datasets import mnist
from numpy.random import seed
from tensorflow import set_random_seed

# seed(1)
# set_random_seed(2)

resnet50_image_size = 224
resnet50_input_shape = (resnet50_image_size, resnet50_image_size, 3)
resnet50_model_name = "resnet50-take1"
resnet50_preprocessing = keras.applications.resnet50.preprocess_input
resnet50_trainable_layers = [
    # "4a_branch",
    #                          "4b_branch",
    #                          "4c_branch",
    #                          "4d_branch",
    #                          "4e_branch",
    #                          "4f_branch",

    # "5a_branch",
    # "5b_branch",
    # "5c_branch",

    "tail"]

xception_image_size = 299
xception_input_shape = (xception_image_size, xception_image_size, 3)
xception_model_name = "xception-stanford-images"
xception_preprocessing = keras.applications.xception.preprocess_input
xception_trainable_layers = ["block13",
                             "block14",
                             "tail"]

ensemble_name = "ensemble"
ensemble_trainable_layers = ["ensemble"]

current_image_size = xception_image_size
current_preprocessing = xception_preprocessing
current_input_shape = xception_input_shape

batch_size = 16
classes_count = 120

input_folder = "../input"
output_dir = "../output"
augmented_dir = output_dir + "/augmented"
augmented_dir = None
test_dir = input_folder + "/test"
# train_dir = input_folder + "/sorted/train"
train_dir = input_folder + "/stanford-images-train-val"
# val_dir = input_folder + "/sorted/validate"
val_dir = input_folder + "/stanford-images-train-val"
submission_dir = output_dir + "/submissions"
shuffling = True

train_datagen = ImageDataGenerator(preprocessing_function=current_preprocessing,
                                   validation_split=0.12,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   rotation_range=45,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    shuffle=shuffling,
                                                    target_size=(current_image_size, current_image_size),
                                                    batch_size=batch_size,
                                                    save_to_dir=augmented_dir,
                                                    subset="training",
                                                    seed=42)

val_datagen = ImageDataGenerator(preprocessing_function=current_preprocessing,
                                 validation_split=0.12)
validation_generator = val_datagen.flow_from_directory(val_dir,
                                                       shuffle=False,
                                                       target_size=(current_image_size, current_image_size),
                                                       batch_size=batch_size,
                                                       subset="validation",
                                                       seed=42)


def is_trainable_layer(list_of_names, layer):
    if list_of_names is None:
        return False
    return any([name for name in list_of_names if (name in layer.name)])


def build_ripped_xception1(output_classes_count, trainable_layers, name, input_shape, dropout=0.2):
    m = xception.Xception(include_top=False, input_shape=input_shape)

    x = GlobalAveragePooling2D(name="tail_avg_pool")(m.output)

    x = Dense(512, activation='relu', name="tail_fc1")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Dense(output_classes_count, name="tail_fc_final", activation='softmax')(x)
    result = Model(m.input, x, name=name)

    for layer in result.layers:
        if not is_trainable_layer(trainable_layers, layer):
            layer.trainable = False

    return result


def load_trained_xception1(input_shape, output_classes_count, name, output_folder, trainable_layers=None):
    model = build_ripped_xception1(output_classes_count=output_classes_count,
                                   trainable_layers=trainable_layers,
                                   name=name,
                                   input_shape=input_shape)
    model.load_weights("{}/weights/{}-weights.hdf5".format(output_folder, name))
    return model


def build_ripped_xception2(output_classes_count, trainable_layers, name, input_shape, dropout=0.2):
    m = xception.Xception(include_top=False, input_shape=input_shape)

    x = GlobalAveragePooling2D(name="tail_avg_pool")(m.output)

    x = Dense(output_classes_count, name="tail_fc_final", activation='softmax')(x)
    result = Model(m.input, x, name=name)

    for layer in result.layers:
        if not is_trainable_layer(trainable_layers, layer):
            layer.trainable = False

    return result


def load_trained_xception2(input_shape, output_classes_count, name, output_folder, trainable_layers=None):
    model = build_ripped_xception2(output_classes_count=output_classes_count,
                                   trainable_layers=trainable_layers,
                                   name=name,
                                   input_shape=input_shape)
    model.load_weights("{}/weights/{}-weights.hdf5".format(output_folder, name))
    return model


def build_ripped_xception3(output_classes_count, trainable_layers, name, input_shape, dropout=0.2):
    m = xception.Xception(include_top=False, input_shape=input_shape)

    x = GlobalAveragePooling2D(name="tail_avg_pool")(m.output)

    x = Dense(output_classes_count, name="tail_fc_final", activation='softmax')(x)
    result = Model(m.input, x, name=name)

    for layer in result.layers:
        if not is_trainable_layer(trainable_layers, layer):
            layer.trainable = False

    return result


def load_trained_xception3(input_shape, output_classes_count, name, output_folder, trainable_layers=None):
    model = build_ripped_xception3(output_classes_count=output_classes_count,
                                   trainable_layers=trainable_layers,
                                   name=name,
                                   input_shape=input_shape)
    model.load_weights("{}/weights/{}-weights.hdf5".format(output_folder, name))
    return model


def build_ripped_resnet50(output_classes_count, trainable_layers, name, input_shape):
    m = resnet50.ResNet50(include_top=False, input_shape=input_shape)

    x = GlobalAveragePooling2D(name="tail_avg_pool")(m.output)

    x = Dense(output_classes_count, name="tail_fc_final", activation='softmax')(x)
    result = Model(m.input, x, name=name)

    for layer in result.layers:
        if not is_trainable_layer(trainable_layers, layer):
            layer.trainable = False

    return result


def load_trained_resnet50(input_shape, output_classes_count, name, output_folder):
    model = build_ripped_resnet50(output_classes_count=output_classes_count,
                                  trainable_layers=None,
                                  name=name,
                                  input_shape=input_shape)
    model.load_weights("{}/weights/{}-weights.hdf5".format(output_folder, name))
    return model


def build_ensemble(xception1,
                   xception2,
                   xception3,
                   input_shape):
    inp = Input(shape=input_shape)
    x1 = xception1(inp)
    x2 = xception2(inp)
    x3 = xception3(inp)

    x = Average()([x1, x2, x3])

    m = Model(inp, x, name="ensemble")
    return m


def predict(trained_model, folder, image_size_w_h, preprocessing_function):
    datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    generator = datagen.flow_from_directory(folder,
                                            shuffle=False,
                                            target_size=image_size_w_h)
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
    df2.to_csv("{}/{}-submission.csv".format(output_folder, model_name), index=None)
    exit(0)


def get_top_predictions(predictions, num_to_class, top=1):
    results = []
    for pred in predictions:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(num_to_class[i], pred[i]) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results


def get_num_to_class(input_folder):
    labels_csv = pd.read_csv(input_folder + "/labels.csv")
    breeds = pd.Series(labels_csv['breed'])
    unique_breeds = np.unique(breeds)
    unique_breeds.sort()

    return dict(zip(range(unique_breeds.size), unique_breeds))


def pandas_classification_report(y_true, y_pred, num_to_class):
    metrics_summary = precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred)

    avg = list(precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index)

    class_report_df = class_report_df.rename(lambda column: num_to_class[column] if column in num_to_class else column,
                                             axis="columns")
    support = class_report_df.loc['support']
    total = support.sum()
    avg[-1] = total

    class_report_df['avg / total'] = avg

    return class_report_df.T


def generate_confision_reports(predictions, input_dir, output_folder, report_name, classes):
    """
    :param predictions:
    :param input_dir:
    :param output_folder:
    :param report_name:
    :param classes: classes from generator (e.g validation_generator.classes)
    """
    y_pred = np.argmax(predictions, axis=1)
    num_to_class = get_num_to_class(input_dir)

    labels = [value for key, value in num_to_class.items()]
    confusion = confusion_matrix([num_to_class[x] for x in classes],
                                 [num_to_class[x] for x in y_pred],
                                 labels=labels)
    c = pd.DataFrame(confusion, index=labels)
    c = c.rename(num_to_class, axis='columns')
    c.to_csv("{}/confusion-{}.csv".format(output_folder, report_name))
    report = pandas_classification_report([num_to_class[x] for x in classes],
                                          [num_to_class[x] for x in y_pred],
                                          num_to_class)
    report.to_csv("{}/classification_report-{}.csv".format(output_folder, report_name))
    return confusion, labels


def plot_confusion(confusion, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# Xception
# p = predict(load_trained_xception1(input_shape=xception_input_shape,
#                                    output_classes_count=classes_count,
#                                    name="xception-first-036",
#                                    output_folder=output_dir),
#             val_dir,
#             image_size_w_h=(current_image_size, current_image_size),
#             preprocessing_function=current_preprocessing)

# create_submission(p,
#                   input_dir,
#                   submission_dir,
#                   xception_model_name)
# print(get_top_predictions(p, get_num_to_class(input_dir)))
# exit(0)
#
model = build_ripped_xception1(output_classes_count=classes_count,
                               trainable_layers=xception_trainable_layers,
                               name=xception_model_name,
                               input_shape=xception_input_shape)
# model = load_trained_xception1(input_shape=xception_input_shape,
#                                output_classes_count=classes_count,
#                                name="xception-best",
#                                output_folder=output_dir,
#                                trainable_layers=xception_trainable_layers)

# ResNet50
# p = predict(load_trained_resnet50(input_shape=resnet50_input_shape,
#                                   output_classes_count=classes_count,
#                                   name=resnet50_model_name,
#                                   output_folder=output_dir),
#             test_dir,
#             image_size_w_h=(current_image_size, current_image_size),
#             preprocessing_function=current_preprocessing)
# create_submission(p,
#                   input_dir,
#                   submission_dir,
#                   resnet50_model_name)
# print(get_top_predictions(p, get_num_to_class(input_dir)))
# exit(0)

# model = build_ripped_resnet50(output_classes_count=classes_count,
#                               trainable_layers=resnet50_trainable_layers,
#                               name=resnet50_model_name,
#                               input_shape=resnet50_input_shape)

# Ensemble

# p = predict(build_ensemble(xception1=load_trained_xception1(input_shape=xception_input_shape,
#                                                             output_classes_count=classes_count,
#                                                             name="xception-first-036",
#                                                             output_folder=output_dir),
#                            xception2=load_trained_xception2(input_shape=xception_input_shape,
#                                                             output_classes_count=classes_count,
#                                                             name="xception-second-044",
#                                                             output_folder=output_dir),
#                            xception3=load_trained_xception3(input_shape=xception_input_shape,
#                                                             output_classes_count=classes_count,
#                                                             name="xception-third-037",
#                                                             output_folder=output_dir),
#                            input_shape=xception_input_shape),
#             test_dir,
#             image_size_w_h=(current_image_size, current_image_size),
#             preprocessing_function=current_preprocessing)
# create_submission(p,
#                   input_dir,
#                   submission_dir,
#                   "ensemble")
# print(get_top_predictions(p, get_num_to_class(input_dir)))
# exit(0)

# model = build_ensemble(output_classes_count=classes_count,
#                        xception=load_trained_xception(input_shape=xception_input_shape,
#                                                       output_classes_count=classes_count,
#                                                       name=xception_model_name,
#                                                       output_folder=output_dir),
#                        xception_input_shape=xception_input_shape,
#                        resnet50=load_trained_resnet50(input_shape=resnet50_input_shape,
#                                                       output_classes_count=classes_count,
#                                                       name=resnet50_model_name,
#                                                       output_folder=output_dir),
#                        resnet50_input_shape=resnet50_input_shape,
#                        trainable_layers=ensemble_trainable_layers)

# learning from scratch
optimizer = keras.optimizers.Adam()
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=2,
                              verbose=1,
                              min_lr=0.0001)

# for fine-tuning
# optimizer = SGD(lr=0.008, decay=1e-6, momentum=0.9, nesterov=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss',
#                               factor=0.2,
#                               patience=2,
#                               verbose=1,
#                               min_lr=0.0001)

model.compile(optimizer=optimizer,
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
weight_saving = ModelCheckpoint(filepath="{}/weights/{}-weights.hdf5".format(output_dir, xception_model_name),
                                verbose=1,
                                save_best_only=True)
callbacks = [early_stopping, weight_saving]
if reduce_lr is not None:
    callbacks.append(reduce_lr)
# callbacks = None

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=int(len(train_generator) / batch_size),
                              epochs=1000,
                              validation_data=validation_generator,
                              validation_steps=int(len(validation_generator) / batch_size),
                              shuffle=shuffling,
                              verbose=2,
                              callbacks=callbacks)
