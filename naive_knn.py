import numpy as np

class KNearestNeighbors:
    """
    Simple implementation of a k-NN estimator.
    """
    def __init__(self, n_neighbors: int = 1) -> None:
        self.k = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Set the train dataset attributes to be used for prediction.
        """
        self.X_train = X_train
        self.y_train = y_train

    def get_neighbor_classes(self, observation: np.ndarray) -> np.ndarray:
        """
        Returns an array of the classes of the *k* nearest neighbors.
        """
        distances = np.sqrt(np.sum((self.X_train - observation)**2, axis=1))

        # Create an array of training set indices ordered by their
        # distance from the current observation
        indices = np.argsort(distances, axis=0)

        selected_indices = indices[:self.k]
        return self.y_train[selected_indices]

    def estimate_class(self, observation: np.ndarray) -> int:
        """
        Estimates to which class a given row (*observation*) belongs.
        """
        neighbor_classes = self.get_neighbor_classes(observation)
        classes, counts = np.unique(neighbor_classes, return_counts=True)
        return classes[np.argmax(counts)]

    # Replace with `mode()`

    def predict(self, X: np.ndarray):
        """
        Apply k-NN estimation for each row in a given dataset.
        """
        return np.apply_along_axis(self.estimate_class, 1, X)
