# -*- coding: utf-8 -*-
"""
================================================================================
 Script de Entrenamiento y Evaluación del Modelo de Riesgo Crediticio
 ================================================================================
 Este script se encarga del preprocesamiento de datos, el entrenamiento de un
 modelo KNN, y la evaluación de su rendimiento utilizando la matriz de
 confusión, F1-score y validación cruzada.
"""

# ==============================================================================
# Importaciones de módulos
# ==============================================================================
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import TomekLinks
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

# Ignorar advertencias para una salida más limpia
warnings.filterwarnings("ignore")

# ==============================================================================
# Carga de datos
# ==============================================================================
file_name = 'financial_risk_assessment.csv'

try:
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, file_name)

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_name)
        except Exception as e:
            print(f"Error al leer el archivo: {e}")
            exit()
    else:
        print(f"\n¡Error! El archivo '{file_path}' NO fue encontrado en el directorio de trabajo actual.")
        print("Asegúrate de que el archivo está en la misma carpeta desde donde estás ejecutando el script.")
        exit()
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")
    exit()

# ==============================================================================
# Preprocesamiento de datos
# ==============================================================================
nombres_es = {
    'Age': 'Edad',
    'Gender': 'Género',
    'Education Level': 'Nivel Educativo',
    'Marital Status': 'Estado Civil',
    'Income': 'Ingreso',
    'Credit Score': 'Puntaje de Crédito',
    'Loan Amount': 'Monto del Préstamo',
    'Loan Purpose': 'Propósito del Préstamo',
    'Employment Status': 'Situación Laboral',
    'Years at Current Job': 'Años en el Empleo Actual',
    'Payment History': 'Historial de Pagos',
    'Debt-to-Income Ratio': 'Relación Deuda-Ingreso',
    'Assets Value': 'Valor de Activos',
    'Number of Dependents': 'Número de Dependientes',
    'City': 'Ciudad',
    'State': 'Estado',
    'Country': 'País',
    'Previous Defaults': 'Incumplimientos Previos',
    'Marital Status Change': 'Cambio de Estado Civil',
    'Risk Rating': 'Calificación de Riesgo'
}

# Renombrar columnas, eliminar nulos y codificar la variable objetivo
df.rename(columns=nombres_es, inplace=True)
df = df.dropna()
df['Calificación de Riesgo'] = df['Calificación de Riesgo'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Definir las columnas de características y la variable objetivo
continuas = ['Ingreso', 'Puntaje de Crédito', 'Monto del Préstamo',
             'Relación Deuda-Ingreso', 'Valor de Activos', 'Incumplimientos Previos']
discretas = ['Edad', 'Años en el Empleo Actual',
             'Número de Dependientes']
categorico = ['Género', 'Nivel Educativo', 'Estado Civil', 'Propósito del Préstamo',
              'Situación Laboral', 'Historial de Pagos', 'Cambio de Estado Civil']

X = df[continuas + discretas + categorico]
y = df['Calificación de Riesgo']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Preprocesamiento de las características con ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('cont', MinMaxScaler(), continuas),
    ('disc', StandardScaler(), discretas),
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorico)
])

# Preprocesar los datos de entrenamiento y prueba
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Obtener los nombres de las columnas después del OneHotEncoding
feature_names = (list(preprocessor.named_transformers_['cont'].get_feature_names_out()) +
                 list(preprocessor.named_transformers_['disc'].get_feature_names_out()) +
                 list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorico)))

# Convertir los datos preprocesados a DataFrame
X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

# ==============================================================================
# Guardar archivos para la aplicación
# ==============================================================================
print("Guardando archivos preprocesados...")
joblib.dump(feature_names, 'nombres_caracteristicas_preprocesadas.joblib')
joblib.dump(X_train_df, 'X_train_riesgo_financiero.joblib')
joblib.dump(X_test_df, 'X_test_riesgo_financiero.joblib')
joblib.dump(y_train, 'y_train_riesgo_financiero.joblib')
joblib.dump(y_test, 'y_test_riesgo_financiero.joblib')
joblib.dump(preprocessor.named_transformers_['cont'], 'scaler_continuas.joblib')
joblib.dump(preprocessor.named_transformers_['disc'], 'scaler_discretas.joblib')
joblib.dump(preprocessor.named_transformers_['cat'], 'encoder_categoricas.joblib')
print("Archivos preprocesados guardados exitosamente.")

# ==============================================================================
# Entrenamiento del modelo
# ==============================================================================
# Definir el pipeline con SMOTETomek y KNN
best_pipeline = make_pipeline(
    StandardScaler(),
    SMOTE(random_state=42),
    TomekLinks(),
    KNeighborsClassifier(n_neighbors=5)
)

print("\nEntrenando el modelo...")
best_pipeline.fit(X_train_df, y_train)

# Guardar el modelo entrenado (el pipeline completo)
joblib.dump(best_pipeline, 'best_pipeline_riesgo_financiero.joblib')
print("Modelo entrenado y guardado exitosamente.")

# ==============================================================================
# Evaluación del modelo
# ==============================================================================
print("\n--- Evaluación del Modelo ---")

# Predecir sobre el conjunto de prueba
y_pred = best_pipeline.predict(X_test_df)

# Matriz de Confusión
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Reporte de Clasificación (incluye F1-score)
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['Bajo', 'Medio', 'Alto']))

# Validación Cruzada (con StratifiedKFold)
print("\n--- Validación Cruzada (5-fold) ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(best_pipeline, X_train_df, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)

print(f"Puntuaciones de F1-macro de validación cruzada: {scores}")
print(f"Puntuación F1-macro promedio: {np.mean(scores):.4f}")
print(f"Desviación estándar de la puntuación: {np.std(scores):.4f}")

print("\nProceso de entrenamiento y evaluación completado. Grupo 4")
