import numpy as np
from typing import List
def clean(x):
    x = np.array(x)
    x = np.round(x, 4)
    x[np.isclose(x, 0)] = 0.0  # kill -0.0 artifacts
    return x.tolist()

class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:

        x = np.array(x)
        W1, W2 = np.array(W1), np.array(W2)
        b1, b2 = np.array(b1), np.array(b2)
        y_true = np.array(y_true)

        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)  

        z2 = W2 @ a1 + b2
        loss = np.mean((z2 - y_true) ** 2)

        n = len(y_true) if y_true.ndim > 0 else 1

        dz2 = (2 * (z2 - y_true)) / n

        dW2 = np.outer(dz2, a1)
        db2 = dz2

        da1 = W2.T @ dz2

        dz1 = da1 * (z1 > 0)

        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {
    "loss": round(float(loss), 4),
    "dW1": clean(dW1),
    "db1": clean(db1),
    "dW2": clean(dW2),
    "db2": clean(db2),
}