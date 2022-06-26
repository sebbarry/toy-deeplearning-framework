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
    # the grad is the gradient value be backprop with
    def backward(self, grad, grad_origin=None): 

        # checks whether we can backprop. of waiting for a gradient in which case decrement the counter.
        if self.autograd: 
            if grad is None: 
                grad = Tensor(np.ones_like(self.data))
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

            if(self.creation_op == "neg"): 
                self.creators[0].backward(self.grad.__neg__())

            if(self.creation_op == "sub"):
                new = Tensor(self.grad.data)
                self.creators[0].backward(new, self)
                new = Tensor(self.grad.__neg__().data)
                self.creators[1].backward(new, self)

            if(self.creation_op == "mul"):
                new = self.grad * self.creators[1]
                self.creators[0].backward(new, self)
                new = self.grad * self.creators[0]
                self.creators[1].backward(new, self)

            # activation functions conditionals.
            if self.creation_op == "sigmoid": 
                ones = Tensor(np.ones_like(self.grad.data))
                self.creators[0].backward(self.grad * (self * (ones - self)))
            if self.creation_op == "tanh":
                ones = Tensor(np.ones_like(self.grad.data))
                self.creators[0].backward(self.grad * (ones - (self * self)))

            if self.creation_op == "relu": #TODO fix this.
                ones = Tensor(np.ones_like(self.grad.data))
                self.creators[0].backward(self.grad * (ones * self))


            # NOTE ?
            if self.creation_op == "index_select":
                new_grad = np.zeros_like(self.creators[0].data)
                indices_ = self.index_select_indices.data.flatten()
                grad_ = grad.data.reshape(len(indices_), -1)
                for i in range(len(indices_)):
                    new_grad[indices_[i]] += grad_[i]
                self.creators[0].backward(Tensor(new_grad))




            # check if layer multiplication w*i
            if(self.creation_op == "mm"):
                # activation function
                act = self.creators[0]
                # weight values
                weights = self.creators[1]
                # perform layer multiplication. the grad value is the delta calculation 
                #       ~ layer_2_delta.dot(weights_1_2.T)
                new = self.grad.mm(weights.transpose())
                # backprop with the activation function on the layer.
                act.backward(new)
                new = self.grad.transpose().mm(act).transpose()
                weights.backward(new)

            if(self.creation_op == "transpose"):
                self.creators[0].backward(self.grad.transpose())



            if("sum" in self.creation_op):
                dim = int(self.creation_op.split("_")[1])
                ds = self.creators[0].data.shape[dim]
                self.creators[0].backward(self.grad.expand(dim,ds))

            if("expand" in self.creation_op):
                dim = int(self.creation_op.split("_")[1])
                self.creators[0].backward(self.grad.sum(dim))



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

    # invert the sign of the Tensor by multiplying the value by -1
    def __neg__(self): 
        if self.autograd:
            return Tensor(self.data * -1, 
                          autograd=True, 
                          creators=[self], 
                          creation_op="neg")
        return Tensor(self.data * -1)


    # subtract values
    def __sub__(self, other): 
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data, 
                          autograd=True, 
                          creators=[self,other], 
                          creation_op="sub")
        return Tensor(self.data - other.data)

    # multiply values
    def __mul__(self, other): 
        if self.autograd and other.autograd: 
            return Tensor(self.data * other.data, 
                          autograd=True, 
                          creators=[self, other],
                          creation_op="mul")
        return Tensor(self.data * other.data)


    # sums accross a dimension of matrices. It scales down a Tensor to a smaller dimension summing 
    # the values along the way.
    def sum(self, dim): 
        if self.autograd: 
            return Tensor(self.data.sum(dim), 
                          autograd=True, 
                          creators=[self],
                          creation_op="sum_"+str(dim))
        return Tensor(self.data.sum(dim))
    

    # expand is called to backprop through a .sum(). function that copies data along a dimension.
    # expand will add a dimension to the Tensor.
    # it will also copy lower dimension arrays into a large data structure of higher dimension.
    def expand(self, dim, copies): 

        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)

        if self.autograd: 
            return Tensor(new_data, 
                          autograd=True, 
                          creators=[self], 
                          creation_op="expand_" + str(dim))
        return Tensor(new_data)

    # transpose a matrix - self explanatory
    def transpose(self): 
        if self.autograd: 
            return Tensor(self.data.transpose(),
                          autograd=True, 
                          creators=[self],
                          creation_op="transpose")
        return Tensor(self.data.transpose())


    # matrix multiplication here
    def mm(self, x):
        if self.autograd:
            return Tensor(self.data.dot(x.data), 
                          autograd=True, 
                          creators=[self,x],
                          creation_op="mm")
        return Tensor(self.data.dot(x.data))



    """
    Activation Functions.
    """
    def sigmoid(self):
        if self.autograd: 
            return Tensor(1 / (1+np.exp(-self.data)),
                          autograd=True, 
                          creators=[self],
                          creation_op="sigmoid")
        return Tensor(1 / (1+np.exp(-self.data)))


    def tanh(self):
        if self.autograd: 
            return Tensor(np.tanh(self.data), 
                                  autograd=True, 
                                  creators=[self],
                                  creation_op="tanh")
        return Tensor(np.tanh(self.data))


    def relu(self):
        if self.autograd: 
            return Tensor(np.maximum(0, self.data), 
                          autograd=True, 
                          creators=[self],
                          creation_op="relu"
                          )
        return Tensor(np.maximum(0, self.data))


    def index_select(self, indices): 
        if select.autograd: 
            new = Tensor(self.data[indices.data], 
                         autograd=True, 
                         creators=[self], 
                         creation_op="index_select")
            nex.index_select_indices=indices
            return new
        return Tensor(self.data[indices.data])


    # class functions
    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())



