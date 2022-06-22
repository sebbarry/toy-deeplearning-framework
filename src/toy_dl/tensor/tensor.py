import numpy as np


"""

A Tensor object allows us to abstract vectors and matrices into one data structure.

--

A matrix is a two-dimensional tensor, and higher dimensions are referred to as n-        dimensional tensors.

---

A tensor contains automatic gradient descent mechanisms. It allows to perform backpropogation on a network automatically.

(note. we first compute the delta values of each layer, take the derivate of each, update the weights accordingly)


"""


class Tensor(object):

    def __init__(self, data, creators=None, creation_op=None):
        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators
        self.grad = None

    # recurse backwards and apply the gradient layer to each parent
    def backward(self, grad): 
        self.grad = grad

        if(self.creation_op == "add"): 
            self.creators[0].backward(grad)
            self.creators[1].backward(grad)

    # when two tensors are added to create a third, 
    # they become the creators of the third tensor. hence the creators list.
    def __add__(self, other):
        return Tensor(self.data + other.data,
                creators=[self,other], 
                creation_op="add")

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())



