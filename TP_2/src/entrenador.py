"""
Este script realiza el entrenamiento de un clasificador de árbol de decisión utilizando un conjunto de datos cargado desde 'hu_dataset.csv'. 
El flujo principal del script es el siguiente:

1. Carga el dataset en un DataFrame de pandas.
2. Separa las características (X) y las etiquetas (y).
3. Divide los datos en conjuntos de entrenamiento y prueba (80%/20%).
4. Entrena un modelo DecisionTreeClassifier con los datos de entrenamiento.
5. Evalúa el modelo utilizando el conjunto de prueba y muestra un reporte de clasificación.
6. Guarda el modelo entrenado en un archivo 'modelo_entrenado.joblib' para su uso posterior.

Dependencias:
- pandas
- scikit-learn
- joblib

Asegúrese de que el archivo 'hu_dataset.csv' esté presente en el mismo directorio que este script.
"""

import pandas as pd # type: ignore
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 
import joblib 

# -------------------- CARGA DEL DATASET --------------------
df = pd.read_csv('hu_dataset.csv', header = 0)

# Se separan características (X) y etiquetas (y)
X = df.iloc[:, :-1]  # Todas las columnas menos la última
y = df.iloc[:, -1]   # Última columna = etiqueta


# -------------------- DIVISIÓN ENTRENAMIENTO/TEST --------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- ENTRENAMIENTO DEL MODELO --------------------

modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

# -------------------- EVALUACIÓN --------------------

y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# -------------------- GUARDAR MODELO ENTRENADO --------------------
joblib.dump(modelo, 'modelo_entrenado.joblib')
print('Modelo guardado como modelo_entrenado.joblib')