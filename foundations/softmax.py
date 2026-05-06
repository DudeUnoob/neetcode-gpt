import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)


        result = []

        maximum = np.max(z)
        
        denominator = 0

        for i in z:

            denominator += np.exp(i - maximum)

            
        
        for i in z:
            
            current_element = (np.exp(i - maximum)) / (denominator)

            result.append(round(current_element, 4))

        
        return result
