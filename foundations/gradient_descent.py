class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places

        def function(x=init):
            return 2 * x

        for i in range(iterations):

            init = init - learning_rate * function(init)

        return round(init, 5)
