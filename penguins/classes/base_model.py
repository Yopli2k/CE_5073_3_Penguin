# _*_ coding: utf-8 _*_
import pickle
import numpy as np
from abc import ABC, abstractmethod

"""
Base class for each of the serialized models.
"""
class BaseModel(ABC):
    def __init__(self, file_name):
        self.file_name = file_name
        self.categorical_cols = ["island", "sex"]
        self.numerical_cols = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
        self._model = None
        self._scaler = None
        self._vectorizer = None

    """
    Make a prediction using the loaded model.
    This method should be implemented by the child class.
    """
    @abstractmethod
    def predict(self, features):
        pass


    """
    Preprocess the features data to be used for prediction.
    """
    def _preprocess(self, features):
        self.__load_model()

        # get features data
        categorical_data = {col: features[col] for col in self.categorical_cols}
        numerical_data = [features[col] for col in self.numerical_cols]

        # transform data for use into prediction calculation
        vectorized_data = self._vectorizer.transform([categorical_data])
        scaled_data = self._scaler.transform([numerical_data])
        return np.hstack((scaled_data, vectorized_data))

    """
    Load the model, scaler and vectorizer from disk if it is not loaded.
    """
    def __load_model(self):
        if self._model is None:
            with open(self.file_name, 'rb') as file:
                self._model, self._scaler, self._vectorizer = pickle.load(file)