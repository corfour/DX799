
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def infer_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Infer likely column names by common variants.
    Returns mapping keys: company, year, emissions, production, avg_price, incident_count.
    """
    cols = {c.lower(): c for c in df.columns}
    def pick(options: List[str]) -> Optional[str]:
        for o in options:
            if o in cols:
                return cols[o]
        return None

    return {
        'company': pick(['company','firm','producer','name','emitter','organization','org','operator']),
        'year': pick(['year','yr','fiscal_year','date_year']),
        'emissions': pick(['emissions','emission','co2','co2e','total_emissions','co2_emissions','tco2e']),
        'production': pick(['production','prod','output','barrels','volume','total_production']),
        'avg_price': pick(['avg_price','price','mean_price','weighted_price','brent','wti','gas_price']),
        'incident_count': pick(['incident_count','incidents','accidents','pipeline_incidents','count']),
    }

def eval_regression(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae}

def nested_cv_estimator(
    estimator,
    param_grid: Dict,
    X: pd.DataFrame,
    y: pd.Series,
    outer_splits: int = 5,
    inner_splits: int = 5,
    random_state: int = 42,
    scoring: str = 'r2'
) -> Dict[str, float]:
    """
    Perform nested cross-validation for a regression estimator.
    Returns mean/std R2 across outer folds and best_params samples.
    """
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    outer_scores = []
    best_params_list = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        gs = GridSearchCV(estimator, param_grid, cv=inner_cv, scoring=scoring, n_jobs=-1)
        gs.fit(X_train, y_train)

        y_hat = gs.best_estimator_.predict(X_test)
        r2 = r2_score(y_test, y_hat)
        outer_scores.append(r2)
        best_params_list.append(gs.best_params_)

    return {
        'outer_mean_R2': float(np.mean(outer_scores)),
        'outer_std_R2': float(np.std(outer_scores)),
        'best_params_samples': best_params_list,
    }
