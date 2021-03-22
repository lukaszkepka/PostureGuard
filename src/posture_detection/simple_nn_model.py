import os
import os.path as path
from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from annotations import Keypoints, SUPPORTED_CLASSES
from posture_detection.preprocessing import PreProcessingPipeline, NormalizePointCoordinatesToBoundingBox, \
    FilterColumns, PointsToVectors


class PostureDetectionModel:

    @abstractmethod
    def preprocess(self, dataset_data_frame: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        pass

    @abstractmethod
    def train(self, train_samples, test_samples, train_labels, test_labels):
        pass

    @abstractmethod
    def evaluate(self, dataset, labels) -> float:
        pass

    @abstractmethod
    def predict(self, dataset: np.ndarray) -> np.ndarray:
        pass


class SimpleNNModel(PostureDetectionModel):
    INPUT_SIZE = len(Keypoints.BODY_POINTS_ATTRIBUTE_NAMES)
    PREPROCESSING_PIPELINE = PreProcessingPipeline([
        FilterColumns(Keypoints.ATTRIBUTE_NAMES),
        NormalizePointCoordinatesToBoundingBox(),
        FilterColumns(Keypoints.BODY_POINTS_ATTRIBUTE_NAMES),
        PointsToVectors(starting_point_name='center')
    ])

    def __init__(self, model_path, load_weights=False):
        self._model = self._create_model()
        self._model_path = model_path
        self._weights_path = os.path.join(self._model_path, 'weights')

        if load_weights:
            self._load_weights()

    def preprocess(self, dataset_data_frame: pd.DataFrame) -> pd.DataFrame:
        return self.PREPROCESSING_PIPELINE.run(dataset_data_frame)

    def train(self, train_samples, test_samples, train_labels, test_labels, epochs=5000):
        if train_samples.shape[1] != self.INPUT_SIZE:
            raise ValueError(f'Invalid training set size, was {train_samples.shape[1]}, should be {self.INPUT_SIZE}')

        if test_samples.shape[1] != self.INPUT_SIZE:
            raise ValueError(f'Invalid test set size, was {test_samples.shape[1]}, should be {self.INPUT_SIZE}')

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        self._model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = self._model.fit(train_samples, train_labels, validation_data=(test_samples, test_labels),
                                  epochs=epochs, callbacks=[callback])

        self._model.save_weights(self._weights_path, overwrite=True)
        self._show_and_save_history(history)

    def evaluate(self, dataset, labels):
        return self._model.evaluate(dataset, labels)

    def predict(self, dataset: pd.DataFrame) -> np.ndarray:
        dataset = self.preprocess(dataset)
        predictions = self._model.predict(dataset)
        return np.round(predictions).flatten().astype('int32')

    def _create_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.INPUT_SIZE),
            tf.keras.layers.Dense(8, activation='sigmoid'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def _load_weights(self):
        self._model.load_weights(self._weights_path)

    def _show_and_save_history(self, history):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.legend()
        plt.savefig(path.join(self._model_path, 'loss'))
        plt.show()

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val accuracy')
        plt.legend()
        plt.savefig(path.join(self._model_path, 'acc'))
        plt.show()
