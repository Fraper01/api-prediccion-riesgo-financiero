# Archivo de la API de Flask (app.py)
# Este script crea una API RESTful con Flask para la predicción de riesgo
# financiero utilizando dos modelos de machine learning: un modelo KNN y
# un Árbol de Decisión. El preprocesamiento de los datos se maneja en el backend
# cargando los objetos preentrenados.

# Importar las bibliotecas necesarias
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)


try:
    # Cargar modelos entrenados (pipelines completos)
    model_knn = joblib.load('best_pipeline_riesgo_financiero.joblib')
    model_arbol = joblib.load('best_pipeline_arbol_decision_riesgo_financiero.joblib')

    # Cargar los objetos de preprocesamiento
    # (escaladores y codificadores)
    scaler_continuas_knn = joblib.load('scaler_continuas.joblib')
    scaler_discretas_knn = joblib.load('scaler_discretas.joblib')
    encoder_categoricas_knn = joblib.load('encoder_categoricas.joblib')
    scaler_continuas_arbol = joblib.load('scaler_continuas_arbol.joblib')
    scaler_discretas_arbol = joblib.load('scaler_discretas_arbol.joblib')
    encoder_categoricas_arbol = joblib.load('encoder_categoricas_arbol.joblib')
    
    print("Ambos modelos y preprocesadores se han cargado exitosamente.")
except FileNotFoundError as e:
    print(f"Error: No se encontró uno de los archivos del modelo o preprocesador. {e}")
    model_knn, model_arbol = None, None
    scaler_continuas_knn, scaler_discretas_knn, encoder_categoricas_knn = None, None, None
    scaler_continuas_arbol, scaler_discretas_arbol, encoder_categoricas_arbol = None, None, None
except Exception as e:
    print(f"Error al cargar los modelos y preprocesadores: {e}")
    model_knn, model_arbol = None, None
    scaler_continuas_knn, scaler_discretas_knn, encoder_categoricas_knn = None, None, None
    scaler_continuas_arbol, scaler_discretas_arbol, encoder_categoricas_arbol = None, None, None


continuas_orig = ['Ingreso', 'Puntaje de Crédito', 'Monto del Préstamo',
                  'Relación Deuda-Ingreso', 'Valor de Activos', 'Incumplimientos Previos']
discretas_orig = ['Edad', 'Años en el Empleo Actual', 'Número de Dependientes']
categorico_orig = ['Género', 'Nivel Educativo', 'Estado Civil', 'Propósito del Préstamo',
                   'Situación Laboral', 'Historial de Pagos', 'Cambio de Estado Civil']
mapeo_riesgo = {0: 'Bajo', 1: 'Medio', 2: 'Alto'}


def preprocesar_datos_knn(df):
    """
    Preprocesa un DataFrame de entrada para el modelo KNN.
    Aplica escalado a variables numéricas y codificación One-Hot a las categóricas.
    """
    df_preprocesado = df.copy()
    df_preprocesado[continuas_orig] = scaler_continuas_knn.transform(df_preprocesado[continuas_orig])
    df_preprocesado[discretas_orig] = scaler_discretas_knn.transform(df_preprocesado[discretas_orig])
    encoded_categoricas = encoder_categoricas_knn.transform(df_preprocesado[categorico_orig])
    encoded_feature_names = encoder_categoricas_knn.get_feature_names_out(categorico_orig)
    df_encoded = pd.DataFrame(encoded_categoricas, columns=encoded_feature_names, index=df_preprocesado.index)
    df_preprocesado = pd.concat([df_preprocesado.drop(categorico_orig, axis=1), df_encoded], axis=1)
    return df_preprocesado

def preprocesar_datos_arbol(df):
    """
    Preprocesa un DataFrame de entrada para el modelo de Árbol de Decisión.
    """
    df_preprocesado = df.copy()
    df_preprocesado[continuas_orig] = scaler_continuas_arbol.transform(df_preprocesado[continuas_orig])
    df_preprocesado[discretas_orig] = scaler_discretas_arbol.transform(df_preprocesado[discretas_orig])
    encoded_categoricas = encoder_categoricas_arbol.transform(df_preprocesado[categorico_orig])
    encoded_feature_names = encoder_categoricas_arbol.get_feature_names_out(categorico_orig)
    df_encoded = pd.DataFrame(encoded_categoricas, columns=encoded_feature_names, index=df_preprocesado.index)
    df_preprocesado = pd.concat([df_preprocesado.drop(categorico_orig, axis=1), df_encoded], axis=1)
    return df_preprocesado


@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    """Realiza la predicción usando el modelo KNN."""
    if model_knn is None:
        return jsonify({'error': 'El Modelo KNN no está disponible.'}), 500

    try:
        data = request.get_json(force=True)
        df_nuevo_solicitante_original = pd.DataFrame([data])
        df_nuevo_solicitante_preprocesado = preprocesar_datos_knn(df_nuevo_solicitante_original)
        prediction_result = model_knn.predict(df_nuevo_solicitante_preprocesado)
        final_prediction = int(prediction_result[0])
        prediction_texto = mapeo_riesgo.get(final_prediction, "Desconocido")
        
        return jsonify({
            'prediction_numerica': final_prediction,
            'prediction_texto': prediction_texto
        })

    except Exception as e:
        return jsonify({'error': f"Error en la predicción del Modelo KNN: {str(e)}"}), 500


@app.route('/predict_arbol', methods=['POST'])
def predict_arbol():
    """Realiza la predicción usando el modelo de Árbol de Decisión."""
    if model_arbol is None:
        return jsonify({'error': 'El Modelo de Árbol de Decisión no está disponible.'}), 500

    try:
        data = request.get_json(force=True)
        df_nuevo_solicitante_original = pd.DataFrame([data])
        df_nuevo_solicitante_preprocesado = preprocesar_datos_arbol(df_nuevo_solicitante_original)
        prediction_result = model_arbol.predict(df_nuevo_solicitante_preprocesado)
        final_prediction = int(prediction_result[0])
        prediction_texto = mapeo_riesgo.get(final_prediction, "Desconocido")
        
        return jsonify({
            'prediction_numerica': final_prediction,
            'prediction_texto': prediction_texto
        })

    except Exception as e:
        return jsonify({'error': f"Error en la predicción del Modelo de Árbol de Decisión: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
