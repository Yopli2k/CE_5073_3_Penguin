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
        probability = self._model.predict_proba(data)[:, 1]
        return {
            "prediction": bool(probability[0] >= 0.5),
            "probability": float(probability[0]),
        }