import argparse
import os
import os.path as path

import cv2
import pandas as pd
import tensorflow as tf

from annotations import SUPPORTED_CLASSES
from posture_detection.simple_nn_model import SimpleNNModel


def parse_args():
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("-i", "--annotations_file_path", required=True,
                    help="Path to annotations in csv format")
    ap.add_argument("-m", "--model_path", required=True,
                    help="Path to trained model")
    return ap.parse_args()


def get_predictions_and_labels(model_path, annotations_data_frame):
    model = SimpleNNModel(model_path, load_weights=True)
    labels = pd.Categorical(annotations_data_frame['class'], categories=SUPPORTED_CLASSES).codes
    predictions = model.predict(annotations_data_frame)
    return predictions, labels


def main(args):
    if not path.exists(args.annotations_file_path):
        print("File {0} doesn't exist".format(args.video_file_path))
        return

    if not path.exists(args.model_path):
        print("Model {0} doesn't exist".format(args.model_path))
        return

    annotations_data_frame = pd.read_csv(args.annotations_file_path)
    predictions, labels = get_predictions_and_labels(args.model_path, annotations_data_frame)

    print('Model {0} evaluation:'.format(args.model_path))
    evaluate_confusion_matrix(labels, predictions)
    evaluate_accuracy(labels, predictions)
    show_invalid_detections(annotations_data_frame, labels, predictions)


def evaluate_confusion_matrix(labels, predictions):
    confusion_matrix = tf.math.confusion_matrix(labels, predictions)
    print('Confusion Matrix = \n {0}'.format(confusion_matrix))


def evaluate_accuracy(labels, predictions):
    accuracy = tf.keras.metrics.Accuracy()
    accuracy.update_state(predictions, labels)
    print('Accuracy = {0:.2f}'.format(accuracy.result().numpy()))


def show_invalid_detections(annotations_data_frame, labels, predictions):
    for index, (cat, pred) in enumerate(zip(labels, predictions)):
        if cat != pred:
            file_path = annotations_data_frame.iloc[index]['file_path']
            im = cv2.imread(file_path)
            text_localisation = (25, 25)
            cv2.putText(im, f'Was {SUPPORTED_CLASSES[int(pred)]} should be {SUPPORTED_CLASSES[int(cat)]}',
                        text_localisation,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        2,
                        cv2.LINE_AA)

            cv2.imshow('Incorrect Detection', im)
            cv2.waitKey(0)


if __name__ == '__main__':
    args = parse_args()
    main(args)
