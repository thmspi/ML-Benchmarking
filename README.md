# Benchmarking Different ML Models

This code allows you to process almost any dataset using data-cleaning methods with just a few parameter tweaks. It then benchmarks 9 different ML models to help you determine which model is best suited for further exploration of your predictions.

## Global Workflow

First, you need to import the dataset you want to work with. You can do this by modifying the following line in `main.py`:

```python
data_path = os.path.join('..', 'data', 'datasets', 'Water Database.xlsx')
data = pd.read_excel(data_path)
```
>**Warning:** Make sure to use the appropriate pandas function depending on the file format you're reading (e.g., read_csv, read_excel, etc.).

Next, the dataset will undergo data cleaning using the `pre_processing(dataframe, target_variable)` function.

## Data Cleaning

This step is the most important and the only one you need to fully understand. Without proper data cleaning, it's nearly impossible for the models to perform well or as expected. You can find all data cleaning related functions in `pre_process.py`

```python
def pre_processing(data :pd.DataFrame, target_var) -> pd.DataFrame:

  data = initial_clean(data, 0.3)
  data = clip_outliers(data, 0.01, 0.99)

#   data = extended_water_clean(data)
#   data = create_daily_lag(data, [target_var], [1, 2, 3], drop_na=True, drop_date_col=False)
#   data = create_hourly_lag(data, [target_var], [1, 2, 6, 12, 24], drop_na=True, drop_date_col=False)
#   data = extract_month_and_season(data, 'Date', add_season=True, drop_date_col=True)
#   data = cyclical_encode_month(data)
  data = factorize_categoricals(data)
  data = scale_numeric(data, 'water_L')

  # Correlation matrix
  correlation_matrix = data.corr()
  plt.figure(figsize=(12, 8))
  sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
  plt.title("Basic")
  plt.show()
  print(data.describe())

  return data
```

The `pre_processing(dataframe, target_variable)` function is composed of multiple helper functions to improve readability and modularity. You can explore the available functions directly in the code. Below is a detailed explanation of each one:

### `data = initial_clean(dataframe, drop_threshold)`

This is the first function to be applied. It performs several important preprocessing tasks:

- Replaces infinite values with `NaN`
- Displays the percentage of missing values per variable
- Converts columns containing `'date'` or `'data'` in their names into `datetime` format
- Drops columns with a missing rate above the given threshold
- Fills other missing values using the **mean** for numerical variables or the **most common value** for categorical ones

---

### `data = clip_outliers(dataframe, low_limit, high_limit)`

This function performs simple outlier detection. It does **not** remove typical outliers—only **extreme values** beyond the specified limits (either very low or very high).

- It will **not** clean all outliers—only the most extreme ones.
- For more advanced outlier handling, consider writing a custom function or manually cleaning the data.
- After trimming extremes, it will display a plot of the data distribution to help visualize any remaining outliers.

---

### `factorize_categoricals(dataframe)`

Machine learning models can't work with categorical variables directly. This function detects categorical variables and **factorizes** them (i.e., converts them into numeric labels).

- This is a **basic encoding method**.
- For better performance or interpretability, you may want to use **one-hot encoding** via `pandas.get_dummies()` if your categorical variables have a limited number of unique values.


Ex : 

| Index | Color |
| ----- | ----- |
| 0     | Red   |
| 1     | Blue  |
| 2     | Green |
| 3     | Blue  |
| 4     | Red   |

Factorizing : 

| Index | Color | Factorized |
| ----- | ----- | ---------- |
| 0     | Red   | 2          |
| 1     | Blue  | 0          |
| 2     | Green | 1          |
| 3     | Blue  | 0          |
| 4     | Red   | 2          |

One Hot Encoding / Get_Dummies() :

| Index | Color | Blue | Green | Red |
| ----- | ----- | ---- | ----- | --- |
| 0     | Red   | 0    | 0     | 1   |
| 1     | Blue  | 1    | 0     | 0   |
| 2     | Green | 0    | 1     | 0   |
| 3     | Blue  | 1    | 0     | 0   |
| 4     | Red   | 0    | 0     | 1   |


### `scale_numeric(dataframe, target_var)`

To achieve better results, it's important to eliminate biases caused by differences in value ranges or scales. This function applies both **normalization** and **standardization** to all numerical variables (which should be the only ones remaining at this stage).

- The `target_var` is excluded from scaling, as it is passed directly from the `pre_processing()` function.
- This ensures the target variable remains on its original scale, which is important for making real-world predictions.

After applying all the functions, the `pre_processing()` function will **plot the correlation matrix** and return the cleaned dataset.

---

#### Global Additional Functions

### `create_daily_lag(dataframe, [variable_list], [lag_days], drop_na, drop_date_col)`

If you need to create daily lags for time series analysis, this function allows you to:

- Generate lagged versions of the variables in `variable_list`
- Specify the number of lag days with `lag_days`
- Optionally drop rows with missing values using `drop_na`
- Optionally drop the original date column using `drop_date_col`

> **Warning:** This function expects a `Date` column to be present in the DataFrame in order to work correctly.

To create lags, you simply need to specify the variable(s) for which you want to create lags, and the time offset (in days). You also have the option to:

- Drop the date column after lag creation (recommended for cleaner datasets)
- Drop rows that do not have lag values (since they would contain `NaN`)

---

### `create_hourly_lag(dataframe, [variable_list], [lag_hours], drop_na, drop_date_col)`

If you need to create **hourly lags**, this function is for you.

> **Warning:** This function expects a `Date` column with **hour-level timestamps** in order to work properly.

- Define the variables to lag via `variable_list`
- Define the number of lag hours via `lag_hours`
- You can choose whether to drop the original date column (`drop_date_col`)
- You can also choose to drop rows with missing lag values (`drop_na`)

---

### `extract_month_and_season(dataframe, date_col, add_season, drop_date_col)`

Use this function to extract **month** and optionally **season** from a date column.

- `date_col`: the name of the date column to use
- `add_season`: set to `True` if you want to extract seasons (encoded as numbers)
- `drop_date_col`: set to `True` to remove the original date column after extraction

> The function expects a valid date column.

---

### `cyclical_encode_month(data)`

To better encode the **month** variable, this function applies **cyclical encoding** using trigonometric functions. This ensures that the model understands the continuity between months (e.g., January is close to December).

> **Warning:** This function expects a column containing `month` in its name. It works automatically if you've used the `extract_month_and_season()` function beforehand.

---

## Specific Additional Functions

### `extended_water_clean(dataframe)`

This function is tailored for Maria’s Water Database and performs advanced outlier correction.

- It identifies sudden drops in data and evaluates the next data point.
  - If the next point is **not** also extreme, it replaces the spike with the **average** of the previous and next values.
  - If the next point is also extreme, it **duplicates** the previous value.

---

### `merge_date_and_hour(dataframe, date_col, hour_col, new_col)`

This function merges separate `date` and `hour` columns into a single `datetime` column.

- `date_col`: name of the date column
- `hour_col`: name of the hour column
- `new_col`: name of the new merged column (if it already exists, it will be overwritten)

> This function is **not** part of the `pre_processing()` function by design, but could be integrated if needed.

---

### `aggregate_building_data(dataframe)`

Designed for use cases where you want to **aggregate water consumption** across multiple buildings.

- Groups data by date and sums water usage across buildings
- Returns a dataset containing **only the date and total water consumption**

> This function is **not** part of the `pre_processing()` function by design, but could be integrated if needed.

---

## Dropping Unwanted Variables

After cleaning your dataset, it's up to you to decide which variables to keep based on their correlation with the target variable. This step is **not automated**, as variable selection requires human judgment.

To drop variables, simply update the following list in the `main.py` file:

```python
data.drop(columns=[], inplace=True)
```

### Creating Training and Testing Datasets

We will now create the training and testing datasets:

- `X_train`: training features  
- `Y_train`: training target variable  
- `X_test`: test features  
- `Y_test`: test target variable (used to evaluate predictions)

---

First, you need to specify which variable is your **target variable** (i.e., the one you want to predict). To do this, modify the value of the following variable:


```python
X = data.drop('water_L', axis=1)
Y = data['water_L']
```

There are two main ways to split your data into training and testing datasets:

---

#### 1. Separating by Time Period (Two Blocks)

If you want to divide the dataset into two distinct **time-based blocks** (e.g., for time series data), you can use the following line of code:

```python
# Keep data sorted by time if it's not already
data = data.sort_index()

# Determine split index
split_index = int(len(data) * 0.9)

# Use the first 90% for training, last 10% for testing
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
Y_train, Y_test = Y.iloc[:split_index], Y.iloc[split_index:]
```

#### 2. Separating Randomly

Often, you may want to evaluate the model’s performance on **random data points** rather than based on time. To do this, you can use the following code:

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 1)
```
### Benchmarking Models

After you've created your training and testing datasets, the code will automatically proceed to evaluate several machine learning models using the provided data.

In the `main.py` script, each model evaluation follows this structure:


```python
RESULTS.append({
    'name': 'KNN',
    'prediction': m.train_knn(X_train, X_test, Y_train, Y_test)
})
```

For each model, we append an element (containing the model name and its predictions) to the `RESULT` list. This list will later be used to **rank the models** based on their performance.

If you want to **add new models**, simply create a new training function in the `models.py` file and follow the same structure as the existing ones.

---

#### Creating Your Own Model Function

The training functions are designed to follow a consistent structure, which looks like this:
```python
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
```

First, we define the model function. For example:

```python
lr = LinearRegression()
```

Then, we define the parameters that we want to test during our search. These parameters will be used to perform a random search for the model.

For example:

```python
param_dist = {
        'fit_intercept': [True, False],
        'positive': [True, False]  # Optional: another param to explore in LinearRegression
    }
```

After that, we use **Random Search** to perform hyperparameter tuning over a broad range of values. 

While this approach won't always find the **optimal** combination, it allows for a much **broader exploration** of the parameter space with lower computational cost compared to Grid Search.

For example:

```python
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
```
After selecting the best parameters, we can proceed to **fit the model**, **export and save** the best model, and **plot the results** obtained during evaluation.


```python
search.fit(X_train, Y_train)
    best = search.best_estimator_

    Y_pred = best.predict(X_test)
    dump(best, os.path.join(model_dir, 'linear_regression.joblib'))
    BEST_MODELS['linear_regression'] = best
    evaluate_model('Linear Regression', Y_test, Y_pred)
```

### Ranking results

Once all models have been trained, the code will display a **ranking of all models** based on their **average error**:

```python
m.rank_models(RESULTS, Y_test)
```

This function compares the predictions stored in the `RESULTS` list and ranks the models accordingly, helping you quickly identify the most effective one.

If available, it will also display the **best parameters** found during hyperparameter tuning — but **only for models** that went through a tuning process (e.g., via `RandomizedSearchCV` or `GridSearchCV`).

