from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from copy import deepcopy
import pandas as pd

def linreg(data: pd.DataFrame, x, y, model, results: dict, col_name: str) -> dict:
    if deepcopy(model).__name__ == 'linregress':
        slope, intercept, r_value, p_value, std_err = model(x=data[x], y=data[y])
        results[col_name] = {
            'model': deepcopy(model),
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value**2,
            'p_value': p_value,
            'std_err': std_err
        }
    return results


def results_func(data: pd.DataFrame, results: dict, params: dict) -> dict:
    model = params['Model']
    col_name = params['col_name']
    X = params['X']
    y = params['y']
    if deepcopy(model).__name__ == 'linregress':
        slope, intercept, r_value, p_value, std_err = model(x=data[X], y=data[y])
        results[col_name] = {
            'model': deepcopy(model),
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value**2,
            'p_value': p_value,
            'std_err': std_err
        }
        return results
    elif deepcopy(model).__name__ == 'OLS':
        results[col_name] = {
            'model': deepcopy(model),
            'params': model.params,
            'rsquared': model.rsquared
        }
        return results
    elif deepcopy(model).__name__ == 'LinearRegression':
        x_x = data[X]
        x_x = pd.get_dummies(x_x, drop_first=True)
        yy = data[y]
        m = model()

        # Split
        xtrain, xtest, ytrain, ytest = train_test_split(x_x, yy, test_size=0.2, random_state=42)

        # Fit
        m.fit(xtrain, ytrain)

        # Prediction
        ypred = m.predict(xtest)

        # Coeff
        coeffs = {'Feature': x_x.columns, 'Coefficient': m.coef_}

        results[col_name] = {
            'model': deepcopy(model),
            'coefficients': coeffs,
            'Intercept': m.intercept_,
            'rsquared': r2_score(ytest, ypred),
            'mae': mean_absolute_error(ytest, ypred),
            'mse': mean_squared_error(ytest, ypred),
            'rmse': root_mean_squared_error(ytest, ypred)
        }
        return results
    elif deepcopy(model).__name__ == 'PolynomialFeatures':
        X = pd.get_dummies(data[X], drop_first=True)
        y = data[y]
        # Split
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

        m = model(degree=params['pol_degree'])
        m.fit(xtrain, ytrain)
        coeffs = m.fit_params
        xtrain_poly = m.transform(xtrain)
        xtest_poly = m.transform(xtest)
        res_params = m.get_params()

        results[col_name] = {
            'model': deepcopy(model),
            'coefficients': coeffs,
            'used_model_parameters': res_params,
            'xtrain_poly': xtrain_poly,
            'xtest_poly': xtest_poly,
            'rsquared': r2_score(ytest, ypred),
            'mae': mean_absolute_error(ytest, ypred),
            'mse': mean_squared_error(ytest, ypred),
            'rmse': root_mean_squared_error(ytest, ypred)
        }
        return results

    else:
        return results
