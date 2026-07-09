import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
    mean_pinball_loss,
)


def build_preprocessor(numeric_cols, categorical_cols, poly_degree=None,
                        interaction_only=False, scale_numeric=False):
    """
    ColumnTransformer factory. Numeric branch optionally gets PolynomialFeatures
    (for the polynomial-regression case) and/or StandardScaler (mandatory for
    Ridge/Lasso/ElasticNet -- unscaled features get penalized unequally by
    L1/L2 regularization purely because of their raw units, which has nothing
    to do with their actual importance).
    Categorical branch is always one-hot, drop-first, unknown-safe.
    """
    numeric_steps = []
    if poly_degree:
        numeric_steps.append(
            ('poly', PolynomialFeatures(degree=poly_degree, interaction_only=interaction_only, include_bias=False))
        )
    if scale_numeric:
        numeric_steps.append(('scale', StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps) if numeric_steps else 'passthrough'

    transformers = []
    if numeric_cols:
        transformers.append(('num', numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols))

    if not transformers:
        raise ValueError("Need at least one of numeric_cols or categorical_cols.")
    return ColumnTransformer(transformers)


def fit_and_score(df, train_idx, test_idx, y_col, model, results, col_name,
                   numeric_cols=None, categorical_cols=None,
                   poly_degree=None, interaction_only=False, scale_numeric=False,
                   log_target=False, quantile_alpha=None):
    """
    One function, every regression type in this notebook goes through it, all
    evaluated on the SAME train_idx/test_idx split. That last part matters:
    comparing models fit on different random splits is not a real comparison.

    quantile_alpha: if set, also computes pinball loss at that quantile
    (does NOT change what the model optimizes -- pass a QuantileRegressor
    already configured with its own quantile/alpha; this just scores it).
    """
    if not (hasattr(model, 'fit') and hasattr(model, 'predict')):
        raise TypeError(f"'{col_name}': Model needs .fit/.predict, got {type(model)}.")

    numeric_cols = numeric_cols or []
    categorical_cols = categorical_cols or []
    all_cols = numeric_cols + categorical_cols

    pre = build_preprocessor(numeric_cols, categorical_cols, poly_degree, interaction_only, scale_numeric)
    pipe = Pipeline([('pre', pre), ('model', model)])

    Xtrain = df.loc[train_idx, all_cols]
    Xtest = df.loc[test_idx, all_cols]
    ytrain = df.loc[train_idx, y_col]
    ytest = df.loc[test_idx, y_col]

    # Fit on log(y) if requested, but ALWAYS score back on the original scale.
    # Comparing RMSE-in-log-rupees against RMSE-in-rupees from other models
    # would be comparing different units and silently invalidate the whole
    # comparison table at the end of this notebook.
    ytrain_fit = np.log1p(ytrain) if log_target else ytrain

    pipe.fit(Xtrain, ytrain_fit)
    pred_raw = pipe.predict(Xtest)
    pred = np.expm1(pred_raw) if log_target else pred_raw

    entry = {
        'model': pipe,
        'features': all_cols,
        'mae': mean_absolute_error(ytest, pred),
        'rmse': root_mean_squared_error(ytest, pred),
        'r2': r2_score(ytest, pred),
        'ytest': ytest,
        'ypred': pred,
    }
    if quantile_alpha is not None:
        entry['pinball_loss'] = mean_pinball_loss(ytest, pred, alpha=quantile_alpha)
        entry['quantile_alpha'] = quantile_alpha

    results[col_name] = entry
    return results


def target_encode_column(df, train_idx, test_idx, col, y_col, smoothing=10):
    """
    Leakage-safe target (mean) encoding: the encoding map is computed from
    TRAIN rows only, then applied to both train and test. Unseen categories
    in test fall back to the global train mean rather than NaN.

    `smoothing` pulls small-count categories toward the global mean instead
    of trusting a raw group mean computed from e.g. 2 rows -- without this,
    a rare category's "mean price" is really just noise dressed up as signal.

    Returns (train_encoded_series, test_encoded_series) -- does NOT mutate df,
    because leaking the encoding into the wrong split is exactly the mistake
    this function exists to prevent.
    """
    train_df = df.loc[train_idx, [col, y_col]]
    global_mean = train_df[y_col].mean()

    group_stats = train_df.groupby(col)[y_col].agg(['mean', 'count'])
    smoothed = (group_stats['mean'] * group_stats['count'] + global_mean * smoothing) / (group_stats['count'] + smoothing)

    train_encoded = df.loc[train_idx, col].map(smoothed).fillna(global_mean)
    test_encoded = df.loc[test_idx, col].map(smoothed).fillna(global_mean)
    return train_encoded, test_encoded
