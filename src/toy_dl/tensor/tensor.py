import np as np


"""
A Tensor object allows us to abstract vectors and matrices into one data structure.
"""

class Tensor(object):

    def __init__(self, data):
        self.data = np.array(data)

    def __add__(self, other):
        return Tensor(self.data + other.data)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data__str__())



