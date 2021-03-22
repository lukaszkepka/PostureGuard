import argparse
import os
import os.path as path

import pandas as pd
from sklearn.model_selection import train_test_split

from annotations import SUPPORTED_CLASSES
from posture_detection.simple_nn_model import SimpleNNModel

MODELS_DIRECTORY = './models'


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

    model = SimpleNNModel(model_path)
    X_train, X_test, Y_train, Y_test = prepare_dataset(model.preprocess, args.annotations_file_path)
    model.train(X_train, X_test, Y_train, Y_test)

    print('Train accuracy = ', model.evaluate(X_train, Y_train))
    print('Test accuracy = ', model.evaluate(X_test, Y_test))


def prepare_dataset(preprocessing_function, annotations_file_path):
    annotations_data_frame = pd.read_csv(annotations_file_path)
    categories = pd.Categorical(annotations_data_frame['class'], categories=SUPPORTED_CLASSES).codes
    preprocessed_dataset = preprocessing_function(annotations_data_frame)
    return train_test_split(preprocessed_dataset, categories, train_size=0.8)


if __name__ == '__main__':
    args = parse_args()
    main(args)
