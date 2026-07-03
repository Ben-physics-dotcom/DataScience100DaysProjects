import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def classification_results_func(data: pd.DataFrame, results: dict, params: dict) -> dict:
    """
    Generic classifier fit/score, dispatched by duck-typing instead of __name__
    string-matching. Any object implementing .fit() / .predict() works here --
    LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier,
    RandomForestClassifier, SVC, etc. -- with NO per-model elif branch required.

    params must contain:
        'X'        : list of feature column names
        'y'        : target column name
        'Model'    : an *instantiated* sklearn classifier, e.g. LogisticRegression(max_iter=1000)
                     (NOT the bare class -- this is the fix for the regression_functions.py
                     limitation where hyperparameters could never be passed in)
        'col_name' : key under which results are stored
        'scale'    : optional bool, default False -- standardize numeric features
                     (matters for KNN / SVC, irrelevant for trees/forests)
        'test_size': optional float, default 0.2
        'random_state': optional int, default 42
    """
    model = params['Model']
    col_name = params['col_name']
    X_cols = params['X']
    y_col = params['y']
    scale = params.get('scale', False)
    test_size = params.get('test_size', 0.2)
    random_state = params.get('random_state', 42)

    if not (hasattr(model, 'fit') and hasattr(model, 'predict')):
        raise TypeError(
            f"'{col_name}': Model must be an instantiated sklearn-API classifier "
            f"(has .fit/.predict). Got {type(model)}."
        )

    X = pd.get_dummies(data[X_cols], drop_first=True)
    y = data[y_col]

    # stratify=y: classification splits MUST preserve class balance,
    # unlike the regression train_test_split calls in regression_functions.py
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if scale:
        scaler = StandardScaler()
        Xtrain = pd.DataFrame(scaler.fit_transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)
        Xtest = pd.DataFrame(scaler.transform(Xtest), columns=Xtest.columns, index=Xtest.index)

    fitted_model = deepcopy(model)
    fitted_model.fit(Xtrain, ytrain)
    ypred = fitted_model.predict(Xtest)

    results[col_name] = {
        'model': fitted_model,
        'features': list(X.columns),
        'accuracy': accuracy_score(ytest, ypred),
        'precision_weighted': precision_score(ytest, ypred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(ytest, ypred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(ytest, ypred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(ytest, ypred, labels=fitted_model.classes_),
        'classes': fitted_model.classes_,
        'classification_report': classification_report(ytest, ypred, zero_division=0),
        'ytest': ytest,
        'ypred': ypred,
    }
    return results
