import numpy as np

def decompose_y(y):
    """
    Décompose les écarts en direction et magnitude.
    
    :param y: Numpy array ou une liste des écarts réels.
    :return: Tuple de deux arrays - direction (1 pour positif, 0 pour négatif), magnitude (valeur absolue de l'écart).
    """
    y_direction = (y > 0).astype(int)
    y_magnitude = np.abs(y)
    return y_direction, y_magnitude

def reconstruct_y(y_direction, y_magnitude):
    """
    Reconstruit les écarts à partir de la direction et de la magnitude.
    
    :param y_direction: Numpy array ou une liste indiquant la direction des écarts.
    :param y_magnitude: Numpy array ou une liste indiquant l'ampleur des écarts.
    :return: Numpy array des écarts reconstruits.
    """
    sign_factors = np.zeros_like(y_direction)  # Initialize with zeros
    sign_factors[y_direction == 1] = 1  # Class 1 remains 1
    sign_factors[y_direction == 0] = -1  # Class 0 becomes -1
    # Unclassified (-1) becomes 0, which is already set by np.zeros_like()

    y_reconstructed = y_magnitude * sign_factors
    return y_reconstructed

def weighted_accuracy_score(y_true_reconstructed, y_pred_simulated):
    """
    Calcule le score de précision pondérée en utilisant les écarts réels et prédits reconstruits.
    
    :param y_true_reconstructed: Numpy array des écarts réels reconstruits.
    :param y_pred_simulated: Numpy array des écarts prédits.
    :return: Score de précision pondérée.
    """
    # Calcul de la direction correcte (1 pour correct, 0 pour incorrect)
    correct_direction = ((y_true_reconstructed * y_pred_simulated) > 0).astype(int)
    
    # Calcul des poids basés sur l'ampleur de l'écart réel
    weights = np.abs(y_true_reconstructed)
    weights /= np.sum(weights)  # Normalisation des poids
    
    # Calcul du score de précision pondérée
    weighted_accuracy = np.sum(correct_direction * weights)
    
    return weighted_accuracy