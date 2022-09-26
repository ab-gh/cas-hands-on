# numpy: our main numerical package
import numpy as np
# matplotlib and seaborn: our plotting packages
import matplotlib.pyplot as plt
import seaborn as sns

# linear algebra and optimisation algorithms
from numpy.linalg import norm
from scipy.optimize import minimize
# some useful package
from copy import deepcopy

from typing import List

class Element:
    def __init__(self):
        self.length = 0
        self.matrix = np.eye(2)

    def __matmul__(self, other):
        return self.matrix @ other

    def __mul__(self, n):
        return [deepcopy(self) for _ in range(n)]

    def __rmul__(self, n): return self * n

class Quadrupole(Element):
    def __init__(self, f, l=0):
        self.matrix = np.array([ [1, 0],
                                 [-1/f, 1] ])
        self.length = l
        self.is_thin = True if l == 0 else False

    def __str__(self):
        return f"<Quadrupole>\n{self.matrix[0]}\n{self.matrix[1]}"

class Drift(Element):
    def __init__(self, l):
        self.matrix = np.array([ [1, l],
                                 [0, 1] ])
        self.length = l
        self.is_thin = False

    def __str__(self):
        return f"<Drift>\n{self.matrix[0]}\n{self.matrix[1]}"

class Lattice:
    def __init__(self, elements: List[Element]):
        self.elements = [elements]
        self.n_elements = len(self.elements)

    def add(self, element: Element):
        self.elements.append(element)
        self.n_elements += 1

    def flatten(self):
        flat = Element()
        for element in self.elements[-1::-1]:
            flat.matrix = element @ flat.matrix
            flat.length = element.length + flat.length
        return flat


    