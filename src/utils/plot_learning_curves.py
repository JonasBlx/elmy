from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(model, X_train, y_train, X_test, y_test, step=10):
    train_errors, test_errors = [], []
    m_values = range(1, len(X_train), step)  # Adjusting the step size for iterations
    for m in m_values:
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_test_predict = model.predict(X_test)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        test_errors.append(mean_squared_error(y_test, y_test_predict))
    plt.plot(m_values, np.sqrt(train_errors), label="Train")
    plt.plot(m_values, np.sqrt(test_errors), label="Test")
    plt.legend()
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.title("Learning Curves")