# In[1]: Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import os
import pre_process as p
import models as m
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# In[2]: Reading the dataset
data_path = os.path.join('..', 'data','datasets', 'Water Database.xlsx')
data = pd.read_excel(data_path)
data.head()

#In[3]: Aggregate the dataset

data = p.merge_date_and_hour(data, date_col='Date', hour_col='Hour', new_col='Date')
data = p.aggregate_building_data(data)
data.head()

#In[4]: Clean the dataset
data = p.pre_processing(data, 'water_L')

# data.drop(columns=[], inplace=True)

#In[5]: Split the dataset into train and evaluation set
X = data.drop('water_L', axis = 1)
Y = data['water_L']

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 1)

# Keep data sorted by time if it's not already
data = data.sort_index()

# Determine split index
split_index = int(len(data) * 0.9)

# Use the first 90% for training, last 10% for testing
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
Y_train, Y_test = Y.iloc[:split_index], Y.iloc[split_index:]

m.train_test_plot(Y_train, Y_test)


#In[6]: Train models
RESULTS = []

#In[7]: Train KNN
RESULTS.append({
    'name': 'KNN',
    'prediction': m.train_knn(X_train, X_test, Y_train, Y_test)
})

#In[8]: Train Linear Regression
RESULTS.append({
    'name': 'Linear Regression',
    'prediction': m.train_linear_regression(X_train, X_test, Y_train, Y_test)
})

#In[9]: Train Polynomial Regression
RESULTS.append({
    'name': 'Polynomial Regression',
    'prediction': m.train_polynomial_regression(X_train, X_test, Y_train, Y_test)
})

#In[10]: Train Random Forest
RESULTS.append({
    'name': 'Random Forest',
    'prediction': m.train_random_forest(X_train, X_test, Y_train, Y_test)
})

#In[11]: Train XGBoost
RESULTS.append({
    'name': 'XGBoost',
    'prediction': m.train_xgboost(X_train, X_test, Y_train, Y_test)
})

#In[12]: Train Tensor flow Neural Network
RESULTS.append({
    'name': 'Neural Network',
    'prediction': m.train_neural_network(X_train, X_test, Y_train, Y_test)
})

#In[13]: Train Tensor flow Neural Network
RESULTS.append({
    'name': 'Exponential Smoothing',
    'prediction': m.train_exponential_smoothing(X_train, X_test, Y_train, Y_test)
})

#In[14]: Train Tensor flow Neural Network
RESULTS.append({
    'name': 'LSTM',
    'prediction': m.train_lstm(X_train, X_test, Y_train, Y_test)
})

#In[15] : Train NN Regressor
RESULTS.append({
    'name': 'MLP Regressor',
    'prediction': m.train_mlp_regressor(X_train, X_test, Y_train, Y_test)
})

#In[15] : Train NN Regressor
RESULTS.append({
    'name': 'MLP Regressor Maria',
    'prediction': m.train_mlp_regressor_2(X_train, X_test, Y_train, Y_test)
})

#In[15] : Train NN Regressor
RESULTS.append({
    'name': 'MLP Regressor Maria 2',
    'prediction': m.train_mlp_regressor_3(X_train, X_test, Y_train, Y_test)
})

#In[15]: Plotting results
m.rank_models(RESULTS, Y_test)

#In[16]: Print best parameters
print(m.get_best_models())
