import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def aggregate_building_data(df):
  aggregation_rules = {
      'water_L': 'sum',
  }
  # Group by lotcode and date
  aggregated_df = df.groupby(['lotcode', 'Date'], as_index=False).agg(aggregation_rules)
  aggregated_df.drop(columns=['lotcode'], inplace = True)
  return aggregated_df

def merge_date_and_hour(df, date_col='date', hour_col='hour', new_col='datetime'):
    """
    Merges a date column and an hour column into a single datetime column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        date_col (str): The name of the column containing the date.
        hour_col (str): The name of the column containing the hour (0–23).
        new_col (str): The name of the new datetime column to create.

    Returns:
        pd.DataFrame: The DataFrame with an additional datetime column.
    """
    df = df.copy()  # avoid modifying original DataFrame
    df[date_col] = pd.to_datetime(df[date_col])
    df[new_col] = df[date_col] + pd.to_timedelta(df[hour_col], unit='h')
    return df

def extended_water_clean(df: pd.DataFrame,
                         water_col: str = 'water_L',
                         date_col: str = 'Date',
                         rel_thresh: float = 0.4) -> pd.DataFrame:
    """
    Detect any sudden jump or drop in `water_col` where the change from the previous
    value exceeds `rel_thresh` * previous_value. Repair those points:
      - If the next point is NOT also flagged, set current = (prev + next) / 2
      - Otherwise, carry forward the previous value.
    """
    data = df.copy()
    # ensure chronological order
    if date_col in data.columns:
        data = data.sort_values(date_col).reset_index(drop=True)
    
    vals = data[water_col].astype(float).to_numpy()
    n = len(vals)
    # precompute which indices are "anomalous"
    anomalous = np.zeros(n, dtype=bool)
    for i in range(1, n):
        prev = vals[i-1]
        # avoid dividing by zero
        if prev != 0 and abs(vals[i] - prev) > rel_thresh * abs(prev):
            anomalous[i] = True
    
    # now repair
    for i in np.where(anomalous)[0]:
        prev = vals[i-1] if i > 0 else np.nan
        nxt = vals[i+1] if i < n-1 else np.nan
        
        # if the very next point is not anomalous & valid, average
        if i < n-1 and not anomalous[i+1] and not np.isnan(nxt) and not np.isnan(prev):
            vals[i] = 0.5 * (prev + nxt)
        # else just carry-forward previous valid
        elif not np.isnan(prev):
            vals[i] = prev
        # otherwise leave it (e.g. first point anomaly)
    
    data[water_col] = vals
    return data

def extract_month_and_season(df: pd.DataFrame,
                             date_col: str,
                             add_season: bool = False,
                             drop_date_col: bool = False) -> pd.DataFrame:
    """
    From `date_col` in `df`, extract:
      - month → new column "month"
      - optionally season → new column "season" (winter/spring/summer/autumn)
    and optionally drop the original date column.
    
    Seasons (Northern hemisphere meteorological):
      • winter: 12, 1, 2
      • spring: 3, 4, 5
      • summer: 6, 7, 8
      • autumn: 9,10,11
    
    Parameters
    ----------
    df           : pd.DataFrame
    date_col     : name of the column to parse as datetime
    add_season   : if True, also add a "season" column
    drop_date_col: if True, drop the original `date_col` from the result
    
    Returns
    -------
    A new DataFrame with the added column(s).
    """
    data = df.copy()
    
    # 1) convert to datetime (coerce errors → NaT)
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    
    # 2) extract month
    data['month'] = data[date_col].dt.month
    
    # 3) optionally map month → season
    if add_season:
        season_map = {
            12:1, 1:1, 2:1,   # winter
             3:2, 4:2, 5:2,   # spring
             6:3, 7:3, 8:3,   # summer
             9:4,10:4,11:4    # autumn
        }
        data['season'] = data['month'].map(season_map)
    
    # 4) optionally drop the original date column
    if drop_date_col:
        data = data.drop(columns=[date_col])
    
    return data

def create_daily_lag(df: pd.DataFrame,
               cols: list[str],
               lags: list[int],
               drop_na: bool = True,
               drop_date_col: bool = True) -> pd.DataFrame:
    data = df.copy()

    # 1) ensure a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'Date' in data.columns:
            data = data.set_index('Date', drop=drop_date_col)
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a 'Date' column")

    data = data.sort_index()

    # 2) generate each lag
    for col in cols:
        if col not in data.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")
        for k in lags:
            lag_name = f"{col}_lag_{k}d"
            # if index has a daily frequency, shift by periods; else align by Timedelta
            if data.index.freq is not None:
                data[lag_name] = data[col].shift(k)
            else:
                data[lag_name] = data[col].shift(freq=pd.Timedelta(days=k))

    # 3) optionally drop rows with any NaN in the new lag columns
    if drop_na:
        lag_cols = [f"{col}_lag_{k}d" for col in cols for k in lags]
        data = data.dropna(subset=lag_cols)

    return data

import pandas as pd

def create_hourly_lag(df: pd.DataFrame,
                      cols: list[str],
                      lags: list[int],
                      drop_na: bool = True,
                      drop_date_col: bool = True) -> pd.DataFrame:

    data = df.copy()

    if not isinstance(data.index, pd.DatetimeIndex):
        if 'Date' in data.columns:
            data = data.set_index('Date', drop=drop_date_col)
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a 'Date' column")

    data = data.sort_index()

    # 2) generate each lag
    for col in cols:
        if col not in data.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")
        for k in lags:
            lag_name = f"{col}_lag_{k}h"
            # if index has a fixed frequency, use periods; otherwise shift by timedelta
            if data.index.freq is not None:
                data[lag_name] = data[col].shift(k)
            else:
                data[lag_name] = data[col].shift(freq=pd.Timedelta(hours=k))

    # 3) optionally drop rows with any NaN in the new lag columns
    if drop_na:
        lag_cols = [f"{col}_lag_{k}h" for col in cols for k in lags]
        data = data.dropna(subset=lag_cols)

    return data


def initial_clean(df: pd.DataFrame, drop_thresh: float = 0.3) -> pd.DataFrame:
    data = df.copy()
    # 1) Replace ±inf with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(100*data.isnull().sum()/len(data))
    # 2) Convert date-like columns
    date_cols = [c for c in data.columns if 'date' in c.lower() or 'data' in c.lower()]
    for c in date_cols:
        data[c] = pd.to_datetime(data[c], errors='coerce')
    # 3) Drop overly-sparse columns
    miss_frac = data.isnull().mean()
    drop_cols = miss_frac[miss_frac > drop_thresh].index.tolist()
    if drop_cols:
        data.drop(columns=drop_cols, inplace=True)
    # 4) Impute remaining missing values
    for c in data.columns:
        if data[c].isnull().any():
            if pd.api.types.is_numeric_dtype(data[c]):
                data[c].fillna(data[c].mean(), inplace=True)
            else:
                data[c].fillna(data[c].mode().iloc[0], inplace=True)
    return data

def clip_outliers(df: pd.DataFrame, low_q: float, high_q: float) -> pd.DataFrame:
    data = df.copy()
    num_cols = data.select_dtypes(include='number').columns
    for c in num_cols:
        lo, hi = data[c].quantile([low_q, high_q])
        data[c] = data[c].clip(lo, hi)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=data['water_L'])
    plt.title('Water_L Outliers')

    plt.tight_layout()
    plt.show()
    return data

def cyclical_encode_month(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    month_cols = [c for c in data.columns if 'month' in c.lower()]
    for c in month_cols:
        m = data[c].astype(int)
        data[f"{c}_sin"] = np.sin(2 * np.pi * m / 12)
        data[f"{c}_cos"] = np.cos(2 * np.pi * m / 12)
        data.drop(columns=[c], inplace=True)
    return data

def factorize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    cat_cols = data.select_dtypes(include=['object', 'category']).columns
    for c in cat_cols:
        data[c] = pd.factorize(data[c])[0]
    return data

def scale_numeric(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
    data = df.copy()
    # select numeric columns, excluding target
    num_cols = [c for c in data.select_dtypes(include='number').columns if c != target_col]
    if num_cols:
        scaler = StandardScaler()
        data[num_cols] = scaler.fit_transform(data[num_cols])
        normalizer = MinMaxScaler()
        data[num_cols] = normalizer.fit_transform(data[num_cols])
    return data

def pre_processing(data :pd.DataFrame, target_var) -> pd.DataFrame:

  data = initial_clean(data, 0.3)
  data = clip_outliers(data, 0.01, 0.99)
#   data = extended_water_clean(data)
#   data = create_daily_lag(data, [target_var], [1, 2, 3], drop_na=True, drop_date_col=False)
#   data = create_hourly_lag(data, [target_var], [1, 2, 6, 12, 24], drop_na=True, drop_date_col=False)
#   data = extract_month_and_season(data, 'Date', add_season=True, drop_date_col=True)
#   data = cyclical_encode_month(data)
  data = factorize_categoricals(data)
  data = scale_numeric(data, target_var)

  # Correlation matrix
  correlation_matrix = data.corr()
  plt.figure(figsize=(12, 8))
  sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
  plt.title("Basic")
  plt.show()
  print(data.describe())

  return data





  