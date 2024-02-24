import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def process_data(dataframe, columns_to_remove=None, lines_to_remove=None, scaler=None):
    df_before = dataframe.copy(deep=True)
    if lines_to_remove:
        dataframe = dataframe.copy(deep = True).drop(index=lines_to_remove, errors='ignore')
    df_after = dataframe.copy(deep=True)
    if "DELIVERY_START" in dataframe.columns:
        dataframe.set_index("DELIVERY_START", inplace=True)
        dataframe.index = pd.to_datetime(dataframe.index)
    original_index = dataframe.index  # Store original index
    if columns_to_remove:
        dataframe.drop(columns=columns_to_remove, axis="columns", errors='ignore', inplace=True)
    means = dataframe.mean()
    dataframe = dataframe.fillna(means)
    if scaler:
        original_columns = dataframe.columns  # Store original column names
        if scaler == "standard":
            scaler_model = StandardScaler()
        elif scaler == "minmax":
            scaler_model = MinMaxScaler()
        else:
            raise ValueError("Unsupported scaler")
        scaled_data = scaler_model.fit_transform(dataframe)
        dataframe = pd.DataFrame(scaled_data, index=original_index, columns=original_columns)  # Convert back to DataFrame
    return dataframe
