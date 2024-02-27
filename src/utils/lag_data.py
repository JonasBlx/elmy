import pandas as pd

def create_lagged_features(X, y, n_lags):
    X_lagged = pd.DataFrame(index=X.index)
    for i in range(n_lags + 1):
        shifted = X.shift(i).add_suffix(f'_lag_{i}')
        X_lagged = pd.concat([X_lagged, shifted], axis=1)
    
    # Remove rows with NaN values
    X_lagged = X_lagged.dropna()
    
    # Align y with the modified index of X_lagged
    # This step ensures that y_lagged contains only the entries that have corresponding entries in X_lagged
    y_lagged = y.reindex(X_lagged.index)

    return X_lagged, y_lagged