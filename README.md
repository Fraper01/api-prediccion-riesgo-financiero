API de Predicción de Riesgo de Préstamo

Este repositorio contiene una API RESTful desarrollada con Flask para predecir el riesgo financiero de un solicitante de préstamo. La API utiliza dos modelos de machine learning previamente entrenados: un modelo de vecinos más cercanos (KNN) y un modelo de árbol de decisión.
Modelos y Preprocesadores

La API requiere varios archivos de modelos y preprocesadores (.joblib) que no se incluyen en el repositorio para mantenerlo ligero y limpio. Estos archivos deben ser generados antes de ejecutar la API.

Para generar los archivos .joblib, necesitas ejecutar los siguientes scripts de Python en el orden correcto:

    entrenamiento_y_evaluacion.py: Este script entrena y guarda el modelo KNN y sus preprocesadores.

    entrenamiento_arbol_decision_con_validacion.py: Este script entrena y guarda el modelo de Árbol de Decisión y sus preprocesadores.

Asegúrate de que tus scripts de entrenamiento generen los siguientes archivos en la misma carpeta que app.py:

    best_pipeline_riesgo_financiero.joblib

    best_pipeline_arbol_decision_riesgo_financiero.joblib

    scaler_continuas.joblib

    scaler_discretas.joblib

    encoder_categoricas.joblib

    scaler_continuas_arbol.joblib

    scaler_discretas_arbol.joblib

    encoder_categoricas_arbol.joblib

Endpoints de la API

La API expone los siguientes endpoints para realizar predicciones:

    POST /predict_knn: Predice el riesgo de préstamo usando el modelo KNN.

    POST /predict_arbol: Predice el riesgo de préstamo usando el modelo de Árbol de Decisión.

Ambos endpoints esperan un JSON en el cuerpo de la solicitud con los datos del solicitante.
Uso

Para usar esta API, sigue estos pasos:

    Clona este repositorio:
    git clone https://github.com/tu-usuario/api-prediccion-riesgo-financiero.git
    cd api-prediccion-riesgo-financiero

    Instala las dependencias necesarias:
    pip install -r requirements.txt

    Genera los archivos de modelos y preprocesadores corriendo los scripts de entrenamiento.

    Ejecuta la aplicación:
    python app.py

La API se ejecutará en http://127.0.0.1:5000.
