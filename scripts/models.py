import os
import numpy as np
import pandas as pd
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.neural_network import MLPRegressor
from scipy.stats import expon


import warnings
warnings.filterwarnings("ignore")

def train_test_plot(train_data, test_data):
    plt.figure(figsize=(16, 4))
    plt.plot(train_data.index, train_data, label="Train", color='blue')
    plt.plot(test_data.index, test_data, label="Test", color='orange')
    plt.legend()
    plt.title("Aigua1_mL Time Series Split")
    plt.xlabel("Date")
    plt.ylabel("Aigua1_mL")
    plt.show()

def ensure_model_dir(path: str = '../data/models/'):
    os.makedirs(path, exist_ok=True)
    return path

BEST_MODELS = {}

parameters = {
  
}

ensure_model_dir('../data/models/')

def get_best_models():
    return BEST_MODELS

def evaluate_model(name, Y_test, Y_pred):
  print (f'========================= {name} evaluation ======================')
  print("Mean Squared Error: ", mean_squared_error(Y_test, Y_pred).round(2))
  print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(Y_test, Y_pred)).round(2))
  print('Mean Absolute Error: ', mean_absolute_error(Y_test, Y_pred).round(2)) 

  # 1) Scatter actual vs predicted
  plt.figure(figsize=(8, 6))
  plt.scatter(Y_test, Y_pred, alpha=0.6)
  plt.plot([Y_test.min(), Y_test.max()],
          [Y_test.min(), Y_test.max()],
          'r--', lw=2)  # 45° line
  plt.xlabel('Actual Total Water Consumption')
  plt.ylabel('Predicted Total Water Consumption')
  plt.title(f'{name} : Actual vs Predicted')
  plt.grid(True)
  plt.tight_layout()
  plt.show()

  # 2) Residual plot
  residuals = Y_test - Y_pred
  plt.figure(figsize=(8, 6))
  plt.scatter(Y_pred, residuals, alpha=0.6)
  plt.hlines(0, Y_pred.min(), Y_pred.max(), linestyles='dashed', colors='r')
  plt.xlabel('Predicted Total Water Consumption')
  plt.ylabel('Residuals (Actual – Predicted)')
  plt.title(f'{name} : Residuals vs Predicted')
  plt.grid(True)
  plt.tight_layout()
  plt.show()

  Y_test_array = np.array(Y_test)
  Y_pred_array = np.array(Y_pred)

  pct_error = np.abs(Y_pred_array - Y_test_array) / np.abs(Y_test_array) * 100
  bins = [0, 2, 5, 10,15,20,30, 50, np.inf]
  labels = ['<2%', '2-5%', '5-10%', '10-15%', '15-20%', '20-30%', '30-50%', '>50%']

  s = pd.Series(pct_error)
  cat = pd.cut(s, bins=bins, labels=labels, right=False)
  counts = cat.value_counts().reindex(labels, fill_value=0).to_dict()
  total = len(s)


  percentages = {label: (counts[label] / total) * 100 for label in labels}
  for label in labels:
      print(f"{label} error: {percentages[label]:.2f}% ({counts[label]} of {total})")

  # 5) Bar plot of residuals over time
  res_series = pd.Series(Y_test - Y_pred, index=Y_test.index)
  plt.figure(figsize=(12, 6))
  res_series.plot(kind='bar')
  plt.xlabel('Date')
  plt.ylabel('Residual (Actual – Predicted)')
  plt.title('Residuals by Date')
  plt.tight_layout()
  plt.show()

  Y_pred = pd.Series(Y_pred, index=Y_test.index)

  plt.figure(figsize=(12, 5))
  plt.plot(Y_test.index, Y_test, label='Actual', color='blue')
  plt.plot(Y_test.index, Y_pred, label='Predicted', color='orange', linestyle='--')
  plt.xlabel('Date')
  plt.ylabel('Total Water Consumption')
  plt.title(f'{name} : Predicted vs Actual')
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()


def rank_models(results, Y_test):
    """
    Compute sum and mean absolute error for each model in results list,
    and return a DataFrame ranked by mean_error ascending.

    Parameters:
    - results: list of dicts {'name':..., 'prediction': np.array}
    - Y_test: pd.Series or np.array of true values (same length/order)

    Returns:
    - pd.DataFrame with columns ['name', 'sum_error', 'mean_error', 'rank']
    """
    y_true = np.array(Y_test)
    records = []
    for entry in results:
        name = entry['name']
        Y_pred = np.array(entry['prediction'])
        errors = np.abs(Y_pred - y_true)
        sum_err = errors.sum()
        mean_err = errors.mean()
        records.append({'name': name, 'sum_error': sum_err, 'mean_error': mean_err})
    df_rank = pd.DataFrame(records)
    df_rank = df_rank.sort_values('mean_error').reset_index(drop=True)
    df_rank['rank'] = df_rank.index + 1
    return df_rank

def train_linear_regression(X_train, X_test, Y_train, Y_test, model_dir = '../data/models/'):

    lr = LinearRegression()
    param_grid = {
        'fit_intercept': [True, False]
    }
    import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump

def train_linear_regression(X_train, X_test, Y_train, Y_test, model_dir='../data/models/'):
    lr = LinearRegression()

    # Random search: define parameter distributions
    param_dist = {
        'fit_intercept': [True, False],
        'positive': [True, False]  # Optional: another param to explore in LinearRegression
    }

    # RandomizedSearchCV setup (light search: max 4 combinations)
    search = RandomizedSearchCV(
        lr,
        param_distributions=param_dist,
        n_iter=4,              # Try 4 random combinations (keeps it fast)
        n_jobs=-1,
        cv=5,                  # Reduce folds to 5 for speed
        verbose=1,
        scoring='neg_mean_squared_error',
        random_state=42
    )

    search.fit(X_train, Y_train)
    best = search.best_estimator_

    Y_pred = best.predict(X_test)
    dump(best, os.path.join(model_dir, 'linear_regression.joblib'))
    BEST_MODELS['linear_regression'] = best
    evaluate_model('Linear Regression', Y_test, Y_pred)
    return Y_pred

def train_polynomial_regression(X_train, X_test, Y_train, Y_test,
                                degrees=[1,2,3,4,5], model_dir = '../data/models/'):
    pipe = Pipeline([
        ('poly', PolynomialFeatures()),
        ('lin', LinearRegression())
    ])
    param_dist = {
        'poly__degree': degrees,
        'lin__fit_intercept': [True, False],
        'lin__positive': [True, False]  # Optional param for LinearRegression
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=min(8, len(degrees) * 4),  # Limit the number of combinations for speed
        n_jobs=-1,
        cv=5,                             # Reduced from 10 to 5 folds for faster results
        verbose=1,
        scoring='neg_mean_squared_error',
        random_state=42
    )
    search.fit(X_train, Y_train)
    best = search.best_estimator_
    Y_pred = best.predict(X_test)
    dump(best, os.path.join(model_dir, 'polynomial_regression.joblib'))
    BEST_MODELS['polynomial_regression'] = best
    evaluate_model('Polynomial Regression', Y_test, Y_pred)
    return Y_pred

def train_knn(X_train, X_test, Y_train, Y_test,
              neighbors=[3, 4, 5], weights=['distance'], model_dir = '../data/models/'):
    param_dist = {
        'n_neighbors': neighbors,
        'weights': weights,
        'algorithm': ['ball_tree', 'kd_tree', 'auto'],  # More flexibility in random search
        'leaf_size': [30, 35, 40, 45, 50],
        'p': [1, 2]  # Manhattan and Euclidean
    }
    knn = KNeighborsRegressor()
    search = RandomizedSearchCV(
        knn,
        param_distributions=param_dist,
        n_iter=10,               # Try 10 combinations max for quick tuning
        n_jobs=-1,
        cv=5,                    # Reduced from 10 to 5 for faster tuning
        verbose=1,
        scoring='neg_mean_squared_error',
        random_state=42
    )
    search.fit(X_train, Y_train)
    best = search.best_estimator_
    Y_pred = best.predict(X_test)
    dump(best, os.path.join(model_dir, 'knn_regression.joblib'))
    BEST_MODELS['knn_regression'] = best
    evaluate_model('KNN', Y_test, Y_pred)
    return Y_pred

def train_random_forest(X_train, X_test, Y_train, Y_test,
                        n_estimators=[400, 500, 600], max_depth=[4, 5, 6], model_dir = '../data/models/'):
    rf = RandomForestRegressor(random_state=42)
    param_dist = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': [8, 10, 12],
        'min_samples_leaf': [3, 4, 5],
        'bootstrap': [True, False]  # Added False for more variation
    }
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=10,               # Try 10 random combinations
        n_jobs=-1,
        cv=5,                    # Reduced folds for speed
        verbose=1,
        scoring='neg_mean_squared_error',
        random_state=42
    )
    search.fit(X_train, Y_train)
    best = search.best_estimator_
    Y_pred = best.predict(X_test)
    dump(best, os.path.join(model_dir, 'random_forest.joblib'))
    BEST_MODELS['random_forest'] = best
    evaluate_model('Random Forest', Y_test, Y_pred)
    return Y_pred

def train_xgboost(X_train, X_test, Y_train, Y_test,
                  n_estimators=[400, 500, 600], learning_rate=[0.005, 0.01, 0.02], max_depth=[2, 3, 4], model_dir = '../data/models/'):
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
    param_dist = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'subsample': [0.9, 1.0],
        'colsample_bytree': [0.9, 1.0],
        'gamma': [0, 0.1, 0.2],               # Optional: regularization term
        'reg_alpha': [0, 0.1, 1],             # L1 regularization
        'reg_lambda': [1, 1.5, 2]             # L2 regularization
    }
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=12,              # Try 12 combinations to balance time and performance
        n_jobs=-1,
        cv=5,                   # Reduce to 5-fold CV for speed
        verbose=1,
        scoring='neg_mean_squared_error',
        random_state=42
    )
    search.fit(X_train, Y_train)
    best = search.best_estimator_
    Y_pred = best.predict(X_test)
    dump(best, os.path.join(model_dir, 'xgboost.joblib'))
    BEST_MODELS['xgboost'] = best
    evaluate_model('XGBoost', Y_test, Y_pred)
    return Y_pred

def build_nn_model(hidden_units=64, learning_rate=1e-3, input_dim=None):
    model = Sequential()
    model.add(Dense(hidden_units, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(hidden_units//2, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_neural_network(X_train, X_test, Y_train, Y_test,
                         epochs=[30, 50, 70] , batch_size=[8, 16, 32], hidden_units=[64, 128, 192], model_dir = '../data/models/'):
    input_dim = X_train.shape[1]
    keras_reg = KerasRegressor(model=build_nn_model,
                               model__input_dim=input_dim,
                               epochs=epochs,
                               batch_size=batch_size,
                               hidden_units=hidden_units,
                               optimizer='adam',
                               loss='mse',
                               verbose=0)
    # RandomizedSearchCV over scikeras params
    param_dist = {
        'hidden_units': hidden_units,
        'epochs': epochs,
        'batch_size': batch_size
    }
    search = RandomizedSearchCV(
        estimator=keras_reg,
        param_distributions=param_dist,
        n_iter=8,                 # Explore 8 random combinations
        cv=3,                     # Reduce CV folds for faster training
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=1,
        random_state=42
    )
    search.fit(X_train, Y_train)
    best = search.best_estimator_
    Y_pred = best.predict(X_test)
    # save the scikeras wrapper which includes the model
    dump(best, os.path.join(model_dir, 'neural_network.joblib'))
    BEST_MODELS['neural_network'] = best
    evaluate_model('Tensor Flow NN', Y_test, Y_pred)
    return Y_pred

def train_exponential_smoothing(X_train, X_test, Y_train, Y_test,  model_dir = '../data/models/'):
    """
    Fit a SARIMAX(0,1,1)x(0,1,1,12) model (equivalent to Holt-Winters) using exogenous X.
    """
    # define SARIMAX to mimic additive Holt-Winters with exogenous regressors
    sarimax = SARIMAX(
        endog=Y_train,
        exog=X_train,
        order=(0,1,1),
        seasonal_order=(0,1,1,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fit = sarimax.fit(disp=False)
    pred = fit.predict(start=len(Y_train), end=len(Y_train)+len(Y_test)-1, exog=X_test)
    Y_pred = pd.Series(pred.values, index=Y_test.index)
    dump(fit, os.path.join(model_dir, 'exp_smoothing_exog.joblib'))
    evaluate_model('ExpSmoothExog', Y_test, Y_pred)
    return Y_pred

def train_lstm(X_train, X_test, Y_train, Y_test, model_dir='../data/models/'):

    n_t = X_train.shape[1]
    X_tr = X_train.values.reshape((-1, n_t, 1))
    X_te = X_test.values.reshape((-1, n_t, 1))

    m = Sequential([
        LSTM(50, activation='relu', input_shape=(n_t, 1)),
        Dense(1)
    ])
    m.compile(optimizer='adam', loss='mse')
    m.fit(X_tr, Y_train, epochs=50, batch_size=32, verbose=0)

    model_path = os.path.join(model_dir, 'lstm.joblib')
    dump(m, model_path)

    Y_pred = m.predict(X_te).flatten()
    evaluate_model('LSTM', Y_test, Y_pred)

    return Y_pred

def train_mlp_regressor(
    X_train, X_test, Y_train, Y_test,
    model_dir='../data/models/'
):
    """
    Grid–search around the best known MLP parameters to fine-tune performance.
    """
    base_mlp = MLPRegressor(random_state=42)

    # Tight grid around alpha=0.01, lr=0.01, layers=(150,100,50), max_iter=1000
    param_dist = {
        'hidden_layer_sizes': [
            (150, 100, 50),
            (200, 100, 50),
            (100, 100, 100)
        ],
        'alpha': [0.001, 0.01, 0.02],
        'learning_rate_init': [0.005, 0.01, 0.02],
        'max_iter': [600, 800, 1000]
    }

    search = RandomizedSearchCV(
        estimator=base_mlp,
        param_distributions=param_dist,
        n_iter=8,                      # Test 8 random combinations for speed
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    search.fit(X_train, Y_train)

    best = search.best_estimator_
    Y_pred = best.predict(X_test)

    # Save & register
    os.makedirs(model_dir, exist_ok=True)
    dump(best, os.path.join(model_dir, 'mlp_regressor.joblib'))
    BEST_MODELS['mlp_regressor'] = best

    # Evaluate
    evaluate_model("MLP Regressor",Y_test, Y_pred)
    return Y_pred


def train_mlp_regressor_2(
    X_train, X_test, Y_train, Y_test,
    model_dir='../data/models/'
):
    """
    RandomizedSearchCV to tune key hyperparameters of MLPRegressor.
    """
    # Base model with default params (others will be tuned)
    size = 1000
    decay = 10 ** -2
    base_mlp = MLPRegressor(hidden_layer_sizes=(size),
                            activation="logistic", # "tanh", "relu"
                            solver="adam", # "lbfgs", "sgd"
                            max_iter=1000, alpha=decay
                            )
    base_mlp.fit(X_train, Y_train)
    Y_pred = base_mlp.predict(X_test)

    # Save and register the model
    os.makedirs(model_dir, exist_ok=True)
    dump(base_mlp, os.path.join(model_dir, 'mlp_regressor.joblib'))
    BEST_MODELS['mlp_regressor'] = base_mlp

    # Evaluate model performance
    evaluate_model("MLP Regressor", Y_test, Y_pred)

    return Y_pred

def train_mlp_regressor_3(
    X_train, X_test, Y_train, Y_test,
    model_dir='../data/models/'
):
    """
    RandomizedSearchCV to tune key hyperparameters of MLPRegressor.
    """
    # Base model with default params (others will be tuned)
    size = 200
    decay = 10 ** -2
    base_mlp = MLPRegressor(
        hidden_layer_sizes=(size, size),
        max_iter=1000,
        activation= "relu",  #"logistic", # "tanh",
        solver="adam", # "lbfgs", "sgd"
        alpha=decay
        )
    base_mlp.fit(X_train, Y_train)
    Y_pred = base_mlp.predict(X_test)

    # Save and register the model
    os.makedirs(model_dir, exist_ok=True)
    dump(base_mlp, os.path.join(model_dir, 'mlp_regressor.joblib'))
    BEST_MODELS['mlp_regressor'] = base_mlp

    # Evaluate model performance
    evaluate_model("MLP Regressor", Y_test, Y_pred)

    return Y_pred

