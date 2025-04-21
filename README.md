# Predicción de Precios de Viviendas (Kaggle)
Este repositorio contiene el código desarrollado para la competición "Home Data for ML Course" de Kaggle, cuyo objetivo es predecir los precios de venta de viviendas en Ames, Iowa.

La solución implementa un pipeline completo de Machine Learning que incluye:
- Carga y exploración de datos inicial.
- Ingeniería de características avanzada.
- Detección y eliminación de outliers.
- Preprocesamiento de datos robusto (imputación, escalado, codificación) usando `sklearn` pipelines.
- Un modelo de ensemble Stacking Regressor combinando XGBoost, LightGBM, CatBoost y Lasso.
- Evaluación del modelo mediante validación cruzada.
- Generación del archivo de submisión (`submission.csv`).

**Resultado en Kaggle:**
Con esta implementación, se alcanzó una posición en el Top 104 (de más de 7000 participantes) en la leaderboard de la competición "Housing Prices Competition for Kaggle Learn Users" de Kaggle, con un score de 12806.03997.

[Enlace a la competición](https://www.kaggle.com/competitions/home-data-for-ml-course/leaderboard#](https://www.kaggle.com/competitions/home-data-for-ml-course/leaderboard)

