import argparse
import os
import os.path as path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from annotations import Keypoints, SUPPORTED_CLASSES
from posture_detection.preprocessing import PreProcessingPipeline, NormalizePointCoordinatesToBoundingBox, \
    KeepColumns, PointsToVectors

MODELS_DIRECTORY = './models'
PREPROCESSING_PIPELINE = PreProcessingPipeline([
    KeepColumns(Keypoints.ATTRIBUTE_NAMES),
    NormalizePointCoordinatesToBoundingBox(),
    KeepColumns(Keypoints.BODY_POINTS_ATTRIBUTE_NAMES),
    PointsToVectors(starting_point_name='center')
])


def parse_args():
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("-i", "--annotations_file_path", required=True,
                    help="Path to annotations in csv format")
    ap.add_argument("-m", "--model_name", required=False,
                    help="Model name", default='default_model')
    return ap.parse_args()


def main(args):
    if not path.exists(args.annotations_file_path):
        print("File {0} doesn't exist".format(args.annotations_file_path))
        return

    model_path = path.join(MODELS_DIRECTORY, args.model_name)
    if not path.exists(model_path):
        os.mkdir(model_path)

    X_train, X_test, Y_train, Y_test = prepare_dataset(args.annotations_file_path)
    N, D = X_train.shape

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(D)),
        tf.keras.layers.Dense(15, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3000)
    model.save(model_path, overwrite=True)

    show_and_save_history(history, model_path)

    print('Train accuracy = ', model.evaluate(X_train, Y_train))
    print('Test accuracy = ', model.evaluate(X_test, Y_test))


def prepare_dataset(annotations_file_path):
    annotations_data_frame = pd.read_csv(annotations_file_path)
    dataset = PREPROCESSING_PIPELINE.run(annotations_data_frame)
    # TODO: move adding category column to preprocessing pipeline
    categories = pd.Categorical(annotations_data_frame['class'], categories=SUPPORTED_CLASSES).codes
    return train_test_split(dataset, categories, train_size=0.8)


def show_and_save_history(history, model_path):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.savefig(path.join(model_path, 'loss'))
    plt.show()

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.legend()
    plt.savefig(path.join(model_path, 'acc'))
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
