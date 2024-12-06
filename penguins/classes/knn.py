# -*- coding: utf-8 -*-
from .base_model import BaseModel

"""
Make a prediction using the loaded KNN model
Returns: { prediction: bool }
"""
class KNNModel(BaseModel):
    def __init__(self):
        super().__init__(
            file_name ='../models/knn.pck'
        )

    def predict(self, features):
        data = self._preprocess(features)
        prediction = self._model.predict(data)[0]  # Predicción directa
        return {
            "prediction": bool(prediction),
            "probability": None  # Don't exist probability for K-NN
        }