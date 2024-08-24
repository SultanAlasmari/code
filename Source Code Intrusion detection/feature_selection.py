from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.constants import Boltzmann

def improved_mi_feature_selection(X, y, k=10):
    """
    Perform feature selection using an improved Mutual Information (MI) method.

    Args:
    - X (array-like): Feature matrix.
    - y (array-like): Target variable.
    - k (int): Number of features to select.

    Returns:
    - selected_features (array): Names of selected features.
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Select top-k features # Based on Boltzmann entropy - Improved
    selector = SelectKBest(mutual_info_classif, k=k, Boltzmann=Boltzmann)
    selector.fit(X_scaled, y)
    selected_feature_indices = selector.get_support(indices=True)

    # Get names of selected features
    selected_features = X[:, selected_feature_indices]

    return selected_features


