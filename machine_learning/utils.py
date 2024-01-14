from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    minmax_scale,
)

def calculate_metrics(df):
    result_metrics = {'mae' : mean_absolute_error(df.value, df.prediction),
                      'rmse' : mean_squared_error(df.value, df.prediction) ** 0.5,
                      'r2' : r2_score(df.value, df.prediction)}
    
    print("Mean Absolute Error:       ", result_metrics["mae"])
    print("Root Mean Squared Error:   ", result_metrics["rmse"])
    print("R^2 Score:                 ", result_metrics["r2"])
    return result_metrics

def watts_to_dbm(watts):
    epsilon = 1e-10  # Small constant to avoid log(0)
    dbm = 10 * np.log10(watts + epsilon) + 30
    return dbm

def dbm_to_watts(dbm):
    epsilon = 1e-10  # Small constant to avoid log(0)
    watts = 10 ** ((dbm - 30) / 10) #- epsilon
    return watts

# Value scaling function for feeding into nn
def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
        "maxabs": MaxAbsScaler(),
        "robust": RobustScaler(),
        "power_yeo-johnson": PowerTransformer(method="yeo-johnson"),
        "power_box-cox": PowerTransformer(method="box-cox"),
        "quantiletransformer-uniform": QuantileTransformer(output_distribution="uniform", random_state=42),
        "quantiletransformer-gaussian": QuantileTransformer(output_distribution="normal", random_state=42),
    }
    return scalers.get(scaler.lower())