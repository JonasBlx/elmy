from sklearn.metrics import make_scorer
import numpy as np

def weighted_accuracy(y_true, y_pred):
    diff = y_pred - y_true
    correct_preds = (np.sign(y_pred) == np.sign(y_true)).astype(int)
    weighted_correct_preds = correct_preds * np.abs(y_true)
    weighted_accuracy = np.sum(weighted_correct_preds) / np.sum(np.abs(y_true))
    return weighted_accuracy

# Create a scorer from the weighted_accuracy function
weighted_accuracy_scorer = make_scorer(weighted_accuracy, greater_is_better=True)