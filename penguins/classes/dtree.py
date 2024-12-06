# -*- coding: utf-8 -*-
from .base_model import BaseModel

"""
Make a prediction using the loaded Decision Tree model
Returns: { prediction: bool }
"""
class DTreeModel(BaseModel):
    def __init__(self):
        super().__init__(
            file_name = '../models/dtree.pck'
        )

    def predict(self, features):
        data = self._preprocess(features)
        prediction = self._model.predict(data)[0]
        return {
            "prediction": bool(prediction),
            "probability": None  # Don't exist probability for Decision Trees
        }