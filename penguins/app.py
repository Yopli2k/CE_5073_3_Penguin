# _*_ coding: utf-8 _*_
from flask import Flask, request, jsonify
from classes.logistic import LogisticModel
from classes.svm import SVMModel
from classes.dtree import DTreeModel
from classes.knn import KNNModel

app = Flask(__name__)

# Global instances of the models
models = {
    "logistic": LogisticModel(),
    "svm": SVMModel(),
    "dtree": DTreeModel(),
    "knn": KNNModel(),
}

"""
Endpoint for predictions.
The parameter determines which model we should use for the prediction.
Valid options: logistic, svm, dtree, knn
"""
@app.route("/<model_name>", methods=["POST"])
def predict(model_name):
    if model_name not in models:
        return jsonify({"error": f"Modelo '{model_name}' no encontrado"}), 404

    model = models[model_name]
    try:
        features = request.json
        result = model.predict(features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error al procesar la solicitud: {str(e)}"}), 400


"""
Run the app server
"""
if __name__ == "__main__":
    app.run(debug=True, port=8000)     # change to False in production
