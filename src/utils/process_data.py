"""
Function to process a dataframe in the elmy project. 
We set the time "DELIVERY_START" as the index, 
we remove selected lines, selected columns, we fill missing data 
and we scale.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def process_data(dataframe, columns_to_remove=None, lines_to_remove=None, scaler=None):
    # Drop lines
    if lines_to_remove:
        dataframe = dataframe.copy(deep = True).drop(index=lines_to_remove, errors='ignore')
    # Set "DELIVERY_START" as the index
    if "DELIVERY_START" in dataframe.columns:
        dataframe.set_index("DELIVERY_START", inplace=True)
        dataframe.index = pd.to_datetime(dataframe.index, utc = True)
    original_index = dataframe.index
    # Drop columns
    if columns_to_remove:
        dataframe.drop(columns=columns_to_remove, axis="columns", errors='ignore', inplace=True)
    means = dataframe.mean()
    # Fill incomplete columns with the mean
    dataframe = dataframe.fillna(means)
    # Scaling data
    if scaler:
        # Store original column names because scaler return arrays
        original_columns = dataframe.columns
        if scaler == "standard":
            scaler_model = StandardScaler()
        elif scaler == "minmax":
            scaler_model = MinMaxScaler()
        else:
            raise ValueError("Unsupported scaler")
        scaled_data = scaler_model.fit_transform(dataframe)
        # Convert back to DataFrame
        dataframe = pd.DataFrame(scaled_data, index=original_index, columns=original_columns)  # Convert back to DataFrame
    return dataframe
