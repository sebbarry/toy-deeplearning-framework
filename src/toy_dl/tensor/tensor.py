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

    def __init__(self, data, 
                 autograd=False,
                 creators=None, 
                 creation_op=None, 
                 id=None):

        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators
        self.grad = None

        self.autograd = autograd
        self.children = {}

        # if the id of this tensor is None, generate one
        if id is None: 
            id = np.random.randint(0,100000)
        self.id = id

        # keeps track of how many children each tensor has.
        if creators is not None: 
            for c in creators: 
                if self.id not in c.children: 
                    c.children[self.id] = 1
                else: 
                    c.children[self.id] += 1

    # checks whether a tensor has received the current number of gradients from each child.
    def all_children_grads_accounted_for(self): 
        for id, cnt in self.children.items(): 
            if cnt != 0: 
                return False
        return True

    

    # recurse backwards and apply the gradient layer to each parent
    # this recurses back to layer_0 tensor 
    def backward(self, grad, grad_origin=None): 

        # checks whether we can backprop. of waiting for a gradient in which case decrement the counter.
        if self.autograd: 
            if grad_origin is not None: 
                if self.children[grad_origin.id] == 0: 
                    raise Exception("cannot back prop more than once")
                else: 
                    self.children[grad_origin.id] -= 1
        
        if self.grad is None: 
            self.grad = grad
        else: 
            self.grad += grad

        if(self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None)):
            if(self.creation_op == "add"): 
                self.creators[0].backward(grad)
                self.creators[1].backward(grad)

    # when two tensors are added to create a third, 
    # they become the creators of the third tensor. hence the creators list.
    def __add__(self, other):
        # if both are autograd, return hirearchical tensor w/ autograd feaures.
        if self.autograd and other.autograd: 
            return Tensor(self.data + other.data,
                    autograd=True,
                    creators=[self,other], 
                    creation_op="add")
        # otherwise return a regular tensor value
        return Tensor(self.data + other.data)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())



