import pandas as pd
import numpy as np
import logging
from time import time
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Loading data...")
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')

        test_ids = test['Id']

        y = train['SalePrice']
        X = train.drop(['SalePrice', 'Id'], axis=1)
        test = test.drop(['Id'], axis=1)

        logger.info("Applying log1p transformation to target variable SalePrice...")
        y = np.log1p(y)

        outlier_indices = [523, 1298]
        logger.info(f"Removing known outliers at indices: {outlier_indices}")
        X = X.drop(outlier_indices, axis=0).reset_index(drop=True)
        y = y.drop(outlier_indices, axis=0).reset_index(drop=True)
        logger.info(f"Data shape after outlier removal: X={X.shape}, y={y.shape}")

        def feature_engineering(df):
            if 'Utilities' not in df.columns:
                 df['Utilities'] = 'AllPub'

            df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF'].fillna(0) + df['GarageArea'].fillna(0)

            df['GrLivArea_OverallQual'] = df['GrLivArea'] * df['OverallQual']
            df['TotalSF_OverallQual'] = df['TotalSF'] * df['OverallQual']

            df['YearBuilt_YearRemodAdd_Diff'] = df['YearRemodAdd'] - df['YearBuilt']
            df['YrSold_YearBuilt_Diff'] = df['YrSold'] - df['YearBuilt']
            df['YrSold_YearRemodAdd_Diff'] = df['YrSold'] - df['YearRemodAdd']

            missing_none_features = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                                     'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
                                     'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                                     'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType']
            df["Lack_of_feature_count"] = df[missing_none_features].isnull().sum(axis=1)
            df["Lack_of_feature_count"] += (df["CentralAir"] == 'N').astype(int)

            df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
            df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
            df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
            df['HasMasVnr'] = (df['MasVnrArea'] > 0).astype(int)
            df['HasWoodDeck'] = (df['WoodDeckSF'] > 0).astype(int)
            df['HasOpenPorch'] = (df['OpenPorchSF'] > 0).astype(int)
            df['HasEnclosedPorch'] = (df['EnclosedPorch'] > 0).astype(int)
            df['Has3SsnPorch'] = (df['3SsnPorch'] > 0).astype(int)
            df['HasScreenPorch'] = (df['ScreenPorch'] > 0).astype(int)
            df['IsNewerDwelling'] = (df['YearBuilt'] > 1999).astype(int)

            df['LotFrontage_LotArea_Ratio'] = df['LotFrontage'].fillna(0) / (df['LotArea'] + 1e-6)
            df['GarageArea_GarageCars_Ratio'] = df['GarageArea'].fillna(0) / (df['GarageCars'].replace(0, 1) + 1e-6)
            df['BsmtFinSF1_BsmtFinSF2_Ratio'] = df['BsmtFinSF1'].fillna(0) / (df['BsmtFinSF2'].fillna(0).replace(0, 1) + 1e-6)

            return df

        logger.info("Applying feature engineering...")
        start_time = time()
        X = feature_engineering(X)
        test = feature_engineering(test)
        logger.info(f"Feature engineering completed in {time()-start_time:.2f}s")

        train_cols = set(X.columns)
        test_cols = set(test.columns)

        missing_in_test = list(train_cols - test_cols)
        for col in missing_in_test:
            logger.warning(f"Column '{col}' present in training data but not in test data. Adding with default value 0.")
            test[col] = 0

        missing_in_train = list(test_cols - train_cols)
        for col in missing_in_train:
             logger.warning(f"Column '{col}' present in test data but not in training data. This should not happen if FE is applied consistently.")
             test = test.drop(col, axis=1)

        all_features = X.columns
        categorical_features = [feature for feature in all_features if X[feature].dtype == "object"]
        numerical_features = [feature for feature in all_features if feature not in categorical_features]

        ordinal_features = {
            'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'],
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

        ordinal_cols = [col for col in ordinal_features.keys() if col in X.columns]
        nominal_features = [feature for feature in categorical_features if feature not in ordinal_cols]

        numerical_features = [col for col in all_features if col not in ordinal_cols and col not in nominal_features]

        logger.info(f"Numerical features: {len(numerical_features)}")
        logger.info(f"Ordinal features: {len(ordinal_cols)}")
        logger.info(f"Nominal features: {len(nominal_features)}")

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
                categories=[ordinal_features[col] for col in ordinal_cols],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numerical_features),
            ('nom', nominal_transformer, nominal_features),
            ('ord', ordinal_transformer, ordinal_cols)
        ], remainder='drop')

        logger.info("Initializing models...")

        xgb = XGBRegressor(
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            eval_metric='rmse',
            reg_alpha=0.005,
            reg_lambda=0.005
        )

        lgbm = LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            reg_alpha=0.005,
            reg_lambda=0.005
        )

        catboost = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.01,
            depth=5,
            random_seed=42,
            verbose=0,
            l2_leaf_reg=3
        )

        lasso = Lasso(alpha=0.0005, max_iter=10000, random_state=42)
        ridge = Ridge(alpha=10.0, random_state=42)

        stacked = StackingRegressor(
            estimators=[
                ('xgb', xgb),
                ('lgbm', lgbm),
                ('catboost', catboost),
                ('lasso', lasso),
                ('ridge', ridge)
            ],
            final_estimator=Ridge(alpha=1.0),
            n_jobs=-1,
            cv=5
        )

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', stacked)
        ])

        logger.info("Starting cross-validation...")
        cv = KFold(n_splits=10, shuffle=True, random_state=42)

        start_time_cv = time()
        cv_scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            error_score='raise'
        )
        logger.info(f"Cross-validation completed in {time()-start_time_cv:.2f}s")

        cv_scores = -cv_scores
        mean_cv_rmse = cv_scores.mean()
        std_cv_rmse = cv_scores.std()
        logger.info(f"RMSE promedio en validación cruzada (log1p): {mean_cv_rmse:.4f} ± {std_cv_rmse:.4f}")

        logger.info("Training final model on full training data...")
        start_time = time()
        model.fit(X, y)
        logger.info(f"Model training completed in {time()-start_time:.2f}s")

        logger.info("Making predictions on test data...")
        test_predictions_log = model.predict(test)

        logger.info("Applying expm1 inverse transformation to predictions...")
        test_predictions = np.expm1(test_predictions_log)

        submission = pd.DataFrame({
            'Id': test_ids,
            'SalePrice': test_predictions
        })

        submission['SalePrice'] = submission['SalePrice'].clip(lower=0)

        submission.to_csv('submission.csv', index=False)
        logger.info("✅ Archivo 'submission.csv' generado con éxito!")

    except FileNotFoundError:
        logger.error("Error: train.csv or test.csv not found. Please ensure they are in the correct directory.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()