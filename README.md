# Aplicación de Inteligencia Artificial a la Elipsometría Espectroscópica

Este repositorio contiene el código desarrollado para el Trabajo de Fin de Grado titulado “Aplicación de algoritmos de Inteligencia Artificial a la elipsometría espectroscópica”, cuyo objetivo es predecir parámetros ópticos de películas delgadas a partir de espectros generados o experimentales utilizando redes neuronales.

## Estructura del proyecto

- `dataset_drude_semi.py`  
  Generación de un dataset de 20.000 espectros simulados con el modelo TLD.

- `drude.py`  
  Implementación del modelo Tauc-Lorentz-Drude: cálculo de $\varepsilon_1$, $\varepsilon_2$, índices $n$, $k$, y espectros $(\Psi, \Delta)$ mediante matrices de transferencia.

- `nn_tld.py`  
  Entrenamiento de una red neuronal tradicional sobre el dataset sintético. Utiliza interpolación espectral, normalización y validación cruzada estratificada. Almacena los modelos y predicciones en Excel.

- `PINN_tld_semi.py`  
  Entrenamiento de una *Physics-Informed Neural Network* (PINN), que incorpora una función de pérdida híbrida: error en parámetros + error en la reconstrucción física del espectro usando el modelo TLD.

- `resultados_tld_semi.py`  
  Carga las predicciones NN y PINN, genera espectros y funciones dieléctricas a partir de los parámetros, compara con espectros reales, y calcula el RMSE por parámetro. Incluye visualizaciones detalladas.

---


## Paquetes

- Python 3.10.0
- TensorFlow
- Pandas, NumPy, Matplotlib, SciPy, Scikit-learn
- Optuna
- `elli.kkr` para validación con Kramers-Kronig
