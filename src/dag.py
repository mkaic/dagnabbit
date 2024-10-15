import numpy as np

class ComputationGraph:
    def __init__(self, num_inputs, num_outputs, num_gates):
        self.permutation = np.range(num_gates)