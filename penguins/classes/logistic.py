# _*_ coding: utf-8 _*_
from .base_model import BaseModel


""" 
Make a prediction using the loaded logistic model
Returns: { prediction: bool, probability: float }
"""
class LogisticModel(BaseModel):
    def __init__(self):
        super().__init__(
            file_name ='../models/logistic.pck'
        )

    def predict(self, features):
        data = self._preprocess(features)
        prediction = self._model.predict(data)[0]
        species = self._species_name(self._model.classes_[prediction])  # Get the species name
        probability = self._model.predict_proba(data)[0, prediction]
        return {
            "species": species,
            "probability": float(probability),
            "percentage": round(probability * 100, 3),
        }