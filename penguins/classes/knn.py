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
        prediction = self._model.predict(data)[0]
        species = self._species_name(self._model.classes_[prediction])  # Get the species name
        return {
            "species": species,
            "probability": 1,  # Don't exist probability for K-Nearest Neighbors
            "percentage": 100.000,
        }