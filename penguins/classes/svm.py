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
        probability = self._model.predict_proba(data)[:, 1]
        return {
            "prediction": bool(probability[0] >= 0.5),
            "probability": float(probability[0]),
        }