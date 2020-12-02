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
        distances = np.sqrt(np.sum((self.X_train - observation) ** 2, axis = 1))

        # Create an array of training set indices ordered by their
        # distance from the current observation
        indices = np.argsort(distances, axis = 0)

        selected_indices = indices[:self.k]
        return self.y_train[selected_indices], distances[selected_indices]

    def estimate_class(self, observation: np.ndarray) -> int:
        """
        Estimates to which class a given row (*observation*) belongs.
        """
        neighbor_classes, _ = self.get_neighbor_classes(observation)
        classes, counts = np.unique(neighbor_classes, return_counts = True)
        return classes[np.argmax(counts)]

    def estimate_class_weighted(self, observation: np.ndarray) -> int:
        """
        Estimates to which class a given row (*observation*) belongs with the weighted prediction.
        """
        neighbor_classes, distances = self.get_neighbor_classes(observation)
        weights =  np.where(distances == 0, 1, 1 / (distances ** 2))
        classes = np.unique(neighbor_classes)
        best_c, best_w = 0, 0
        for c in np.nditer(classes):
            w = np.sum(weights[np.nonzero(neighbor_classes == c)])
            best_w, best_c = (w, c) if w > best_w else (best_w, best_c)
        return best_c

    def predict(self, X: np.ndarray):
        """
        Apply k-NN estimation for each row in a given dataset.
        """
        return np.apply_along_axis(self.estimate_class, 1, X)

    def predict_weighted(self, X: np.ndarray):
        """
        Apply k-NN estimation for each row in a given dataset with weighted predictions.
        """
        return np.apply_along_axis(self.estimate_class_weighted, 1, X)
