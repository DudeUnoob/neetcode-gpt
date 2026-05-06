import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places

        return np.round(X @ weights, 5)
        pass

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        
        total = 0

        for i, j in zip(model_prediction, ground_truth):

            i, j = i[0], j[0]

            total += (i - j) ** 2

        return round((1/ len(model_prediction)) * total, 5)

            
            

