import numpy as np
import sys



from src.toy_dl.tensor.tensor import Tensor

# This is the parent superclass.
class Layer(object):

    def __init__(self):
        self.parameters = list()

    def get_parameters(self):
        return self.parameters



class Linear(Layer):

    def __init__(self, n_inputs, n_outputs):
        # init parent class
        super().__init__() 

        # weight layer.
        W = np.random.randn(n_inputs, n_outputs)*np.sqrt(2.0/(n_inputs)) 

        self.weight = Tensor(W, autograd=True)
        self.bias = Tensor(np.zeros(n_outputs), autograd=True)

        # append to parameters
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)


    # forward prop function multiplying weights and layers.
    def forward(self, input):
        return input.mm(self.weight)+self.bias.expand(0, len(input.data))


"""
This is a sequential layer.
It forward propogates a list of layers, where each layer feeds its outputs into to the inputs of the next layer.
"""
class Sequential(Layer):

    def __init__(self, layers=list()):
        super().__init__()

        self.layers = layers

    def add(self, layer):
        for layer in self.layers: 
            input = layer.forward(input)
        return input

    def forward(self, input):
        for layer in self.layers: 
            input = layer.forward(input)
        return input

    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params


"""
This is a loss function layer.
It is a function on the input activated during the forward propogation.
"""
class MSELoss(Layer):

    def __init__(self):
        super().__init__()


    def forward(self, pred, target):
        return ((pred-target)*(pred-target)).sum(0)



class Tanh(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.tanh()


class Sigmoid(Layer):
    
    def __init__(self):
        return super().__init__()

    def forward(self, input):
        return input.sigmoid()


class Relu(Layer):

    def __init__(self):
        return super().__init__()

    def forward(self, input):
        return input.relu()



"""
Embedding layer for layer mapping to words. 
200 words ~ 200 embeddings.
"""
class Embedding(Layer):

    def __init__(self, vocab_size, dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        weight = np.random.random((vocab_size, dim) - 0.5) / dim # convention from word2vec
        self.weight = Tensor(weight, autograd=True)

        self.parameters.append(self.weight)

    def forward(self, input): 
        return self.weight.index_select(input)

        

