import argparse
import os.path as path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from annotations import Keypoints, SUPPORTED_CLASSES
from posture_detection.preprocessing import PreProcessingPipeline, NormalizePointCoordinatesToBoundingBox, \
    KeepColumns, PointsToVectors


def parse_args():
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("-i", "--annotations_file_path", required=True,
                    help="Path to annotations in csv format")
    return ap.parse_args()


def main(args):
    if not path.exists(args.annotations_file_path):
        print("File () doesn't exist".format(args.video_file_path))
        return

    annotations_data_frame = pd.read_csv(args.annotations_file_path)
    preprocessing_pipeline = PreProcessingPipeline([
        KeepColumns(Keypoints.ATTRIBUTE_NAMES),
        NormalizePointCoordinatesToBoundingBox(),
        KeepColumns(Keypoints.BODY_POINTS_ATTRIBUTE_NAMES),
        PointsToVectors(starting_point_name='center')
    ])

    dataset = preprocessing_pipeline.run(annotations_data_frame)
    # TODO: move adding category column to preprocessing pipeline
    categories = pd.Categorical(annotations_data_frame['class'], categories=SUPPORTED_CLASSES).codes

    X_train, X_test, Y_train, Y_test = train_test_split(dataset, categories, train_size=0.9)
    N, D = X_train.shape

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(D)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1000)

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.show()

    print('Train accuracy = ', model.evaluate(X_train, Y_train))
    print('Test accuracy = ', model.evaluate(X_test, Y_test))


if __name__ == '__main__':
    args = parse_args()
    main(args)
