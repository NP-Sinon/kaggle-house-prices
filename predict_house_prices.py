# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fHDSq0zvwI0gk1e-3SXbiYq28W2LXNoC
"""

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# Carga de los datos de entrenamiento y prueba
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Definición de la variable objetivo y las características
y = train['SalePrice']
X = train.drop(['SalePrice'], axis=1)

def feature_engineering(df):
    """
    Realiza la ingeniería de características en el dataframe proporcionado.

    Args:
        df (pd.DataFrame): El dataframe al que se aplicará la ingeniería de características.

    Returns:
        pd.DataFrame: El dataframe con las nuevas características añadidas.
    """
    df["Lack_of_feature_index"] = df[["Street", "Alley", "MasVnrType", "GarageType", "MiscFeature", 'BsmtQual',
                                        'FireplaceQu','PoolQC','Fence']].isnull().sum(axis=1) + (df["MasVnrType"] == 'None') + (df["CentralAir"] == 'No')
    df["MiscFeatureExtended"] = (df["PoolQC"].notnull()*1 + df["MiscFeature"].notnull()*1 + df["Fence"].notnull()*1).astype('int64')
    df["Has_Alley"] = df["Alley"].notnull().astype('int64')
    df["Lot_occupation"] = df["GrLivArea"] / df["LotArea"]
    df["Number_of_floors"] = ((df["TotalBsmtSF"] != 0).astype('int64') +
                                (df["1stFlrSF"] != 0).astype('int64') +
                                (df["2ndFlrSF"] != 0).astype('int64'))
    df['Total_Close_Live_Area'] = df['GrLivArea'] + df['TotalBsmtSF']
    df['Outside_live_area'] = (df['WoodDeckSF'] + df['OpenPorchSF'] +
                                 df['EnclosedPorch'] + df['3SsnPorch'] +
                                 df['ScreenPorch'])
    df['Total_usable_area'] = df['Total_Close_Live_Area'] + df['Outside_live_area']
    df['Area_Quality_Indicator'] = df['Total_usable_area'] * df['OverallQual']
    df['Area_Qual_Cond_Indicator'] = df['Total_usable_area'] * df['OverallQual'] * df['OverallCond']
    df['TotalBath'] = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
    df["Has_garage"] = df["GarageYrBlt"].notnull().astype('int64')
    df['House_Age'] = df['YrSold'] - df['YearBuilt']
    df["Is_Remodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype('int64')
    df['HasBsmt'] = df['BsmtQual'].notnull().astype('int64')
    df['Quality_conditition'] = df['OverallQual'] * df['OverallCond']
    df['Quality_conditition_2'] = df['OverallQual'] + df['OverallCond']
    df['House_Age2'] = df['YrSold'] - df['YearRemodAdd']
    return df

# Aplicación de la ingeniería de características a los conjuntos de entrenamiento y prueba
X = feature_engineering(X)
test = feature_engineering(test)

# Identificación y eliminación de valores atípicos basados en ciertas características y el precio de venta
outliers = X[
    ((X['GrLivArea'] > 4000) & (y < 200000)) |
    ((X['GarageArea'] > 1200) & (y < 300000)) |
    ((X['TotalBsmtSF'] > 4000) & (y < 200000)) |
    ((X['1stFlrSF'] > 4000) & (y < 200000)) |
    ((X['TotRmsAbvGrd'] > 12) & (y < 230000))
].index

X = X.drop(outliers)
y = y.drop(outliers)

# Separación de las características por tipo para aplicar diferentes preprocesamientos
categorical_features = [feature for feature in X.columns if X[feature].dtype == "object"]

# Definición del orden de las categorías para las características ordinales
ordinal_features = {
    'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'],
    'Utilities': ['AllPub', 'NoSewr', 'NoSeWa', 'ELO'],
    'LandSlope': ['Gtl', 'Mod', 'Sev'],
    'ExterQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'ExterCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'BsmtQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'BsmtCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'BsmtExposure': ['Gd', 'Av', 'Mn', 'No', 'NA'],
    'BsmtFinType1': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'],
    'BsmtFinType2': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'],
    'HeatingQC': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'KitchenQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'Functional': ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'],
    'FireplaceQu': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'GarageFinish': ['Fin', 'RFn', 'Unf', 'NA'],
    'GarageQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'GarageCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'PavedDrive': ['Y', 'P', 'N'],
    'PoolQC': ['Ex', 'Gd', 'TA', 'Fa', 'NA'],
    'Fence': ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA']
}

# Separación de características nominales y numéricas
nominal_features = [feature for feature in categorical_features if feature not in ordinal_features]
numerical_features = [feature for feature in X.columns if feature not in categorical_features]

# Identificación de características numéricas discretas y continuas (aunque no se usan explícitamente en transformadores separados)
discrete_numerical_features = [
    'OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath',
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
    'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MSSubClass',
    'Lack_of_feature_index', 'MiscFeatureExtended', 'Has_Alley',
    'Number_of_floors', 'Has_garage', 'Is_Remodeled', 'HasBsmt',
    'Quality_conditition_2'
]

continuous_numerical_features = [
    'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd',
    'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
    'GrLivArea', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
    'PoolArea', 'MiscVal', 'Lot_occupation', 'Total_Close_Live_Area',
    'Outside_live_area', 'Total_usable_area', 'Area_Quality_Indicator',
    'House_Age', 'Area_Qual_Cond_Indicator', 'Quality_conditition',
    'House_Age2', 'TotalBath'
]

# Definición de los transformadores para cada tipo de característica
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(
        categories=[ordinal_features[col] for col in ordinal_features],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ))
])

# Creación del preprocesador utilizando ColumnTransformer para aplicar los transformadores a las columnas correctas
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_features),
    ('nom', nominal_transformer, nominal_features),
    ('ord', ordinal_transformer, list(ordinal_features.keys()))
])

# Definición de los modelos base para el ensamble stacking
xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42
)

lgbm = LGBMRegressor(
    colsample_bytree=0.8,
    learning_rate=0.05,
    max_depth=4,
    n_estimators=1000,
    subsample=0.8,
    random_state=42
)

catboost = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.01,
    depth=6,
    l2_leaf_reg=5,
    verbose=0,
    random_state=42
)

lasso = Lasso(alpha=0.01, max_iter=100000, random_state=42)

# Creación del modelo de stacking que combina los modelos base con un regresor Ridge como estimador final
stacked = StackingRegressor(
    estimators=[
        ('xgb', xgb),
        ('lgbm', lgbm),
        ('catboost', catboost),
        ('lasso', lasso)
    ],
    final_estimator=Ridge(),
    n_jobs=-1
)

# Creación del pipeline completo que incluye el preprocesador y el modelo de regresión stacking
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', stacked)
])

# Evaluación del modelo mediante validación cruzada
cv_scores = cross_val_score(
    model, X, y,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
cv_scores = -cv_scores
mean_cv_mse = cv_scores.mean()
print(f"Promedio del Error Cuadrático Medio (MSE) en la validación cruzada: {mean_cv_mse:.4f}")

# Entrenamiento del modelo final con todos los datos de entrenamiento
model.fit(X, y)

# Realización de predicciones en el conjunto de prueba
test_predictions = model.predict(test)

# Creación del dataframe de envío con los IDs y las predicciones de SalePrice
submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': test_predictions
})

# Guardado del dataframe de envío en un archivo CSV
submission.to_csv('submission.csv', index=False)
print("Archivo 'submission.csv' generado con éxito!")
print(f"- El promedio del MSE estimado a través de la validación cruzada es de: {mean_cv_mse:.4f}")