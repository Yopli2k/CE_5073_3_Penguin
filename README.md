# Penguin Classifier API

Proyecto que implementa una API para la clasificación de pingüinos.
Se basa en modelos de machine learning y utiliza datos de las características físicas de los pingüinos 
como la longitud y profundidad del culmen, la longitud de las aletas, su masa corporal y el sexo 
para predecir su especie.

## Descripción

Esta API se basa en Flask y permite clasificar pingüinos en las especies:
- **Adelie**
- **Chinstrap**
- **Gentoo**

## Características

- **Modelos soportados**:
  - Regresión Logística
  - Máquinas de Soporte Vectorial (SVM)
  - Árboles de Decisión
  - K-Nearest Neighbors (KNN)

- **Entrada JSON**: La API recibe las características de un pingüino en formato JSON.
- **Respuesta JSON**: La API devuelve la predicción de la especie del pingüino.
