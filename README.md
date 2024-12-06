# Penguin Classifier API

Proyecto que implementa una API para la clasificación de pingüinos.  
Se basa en modelos de machine learning y utiliza datos de las características físicas de los pingüinos como:  
- Longitud y profundidad del culmen  
- Longitud de las aletas  
- Masa corporal  
- Sexo  

La API predice la especie del pingüino basándose en estos datos.

---

## Descripción

Esta API utiliza Flask para proporcionar predicciones sobre las especies de pingüinos. Las especies soportadas son:  
- **Adelie**  
- **Chinstrap**  
- **Gentoo**

---

## Características

- **Modelos soportados**:
  - Regresión Logística
  - Máquinas de Soporte Vectorial (SVM)
  - Árboles de Decisión
  - K-Nearest Neighbors (KNN)

- **Entrada JSON**: La API recibe las características de un pingüino en formato JSON.  
- **Respuesta JSON**: La API devuelve la predicción de la especie del pingüino.

---

## Rutas de la API

| Método | Ruta             | Descripción                                |
|--------|------------------|--------------------------------------------|
| POST   | `/logistic`      | Predicción con el modelo de Regresión Logística |
| POST   | `/svm`           | Predicción con el modelo SVM              |
| POST   | `/decision_tree` | Predicción con el modelo Árbol de Decisión |
| POST   | `/knn`           | Predicción con el modelo KNN              |

---

## Descripción de Carpetas

**datasets**: Contiene los datos de los pingüinos usados para entrenar los modelos.

**models**: Contiene los modelos serializados en formato .pck, listos para ser usados en la API.

**notebooks**: Incluye cuadernos Jupyter que contienen:
- El código usado para entrenar y serializar los modelos.
- Código para probar las llamadas de la API.

**penguins**: Carpeta principal de la API en Python.

**penguins/classes**: implementaciones de los modelos que realizan las predicciones.

