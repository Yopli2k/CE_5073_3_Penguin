# -*- coding: utf-8 -*-
from .base_model import BaseModel


"""
Make a prediction using the loaded SVM model
Returns: { prediction: bool, probability: float }
"""
class SVMModel(BaseModel):
    def __init__(self):
        super().__init__(
            file_name ='../models/svm.pck'
        )

    def predict(self, features):
        data = self._preprocess(features)
        prediction = self._model.predict(data)[0]
        species = self._species_name(self._model.classes_[prediction])  # Get the species name
        probability = self._model.predict_proba(data)[0, prediction]
        return {
            "species": species,
            "probability": float(probability),
            "percentage": round(probability * 100, 3)
        }