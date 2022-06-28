import numpy as np
import sys



from toy_dl.tensor.tensor import Tensor

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
These are loss function layers.
It is a way to determine loss accross the layer through a forward propogation using the prediction and the target values..

There are two different loss functions defined below. 
"""
class MSELoss(Layer):

    def __init__(self):
        super().__init__()


    def forward(self, pred, target):
        return ((pred-target)*(pred-target)).sum(0)


# Cross Entropy Loss Layer
class CrossEntropyLoss(Layer):

    def __init__(self):
        super().__init__()

    # TODO move xent loss logic from Tensor Class here? 
    def forward(self, input, target): 
        return input.cross_entropy(target)


"""
Activation Function Layers
"""
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

        weight = np.random.rand(vocab_size, dim) - 0.5 / dim # convention from word2vec
        self.weight = Tensor(weight, autograd=True)

        self.parameters.append(self.weight)

    def forward(self, input): 
        return self.weight.index_select(input)

        

""" 
RNN Layer 
"""
# this is constructed using three linar layers and the .forward() method to take both the output from the previous hidden state
# and the input from the current training data.

class RNNCell(Layer): 

    def __init__(self, n_inputs, n_hidden, n_output, activation="sigmoid"): 
        super().__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output

        if activation == "sigmoid":
            self.activation = Sigmoid()
        elif activation == "tanh":
            self.activation = Tanh()
        elif activation == "relu":
            self.activation = Relu()
        else: 
            raise Exception("Non Linerity not found")

        self.w_ih = Linear(n_inputs, n_hidden)
        self.w_hh = Linear(n_hidden, n_hidden0)
        self.w_ho = Linear(n_hidden, n_output)

        self.parameters += self.w_ih.get_parameters()
        self.parameters += self.w_hh.get_parameters()
        self.parameters += self.w_ho.get_parameters()

    def forward(self, input, hidden):
        from_prev_hidden = self.w_hh.forward(hidden)
        combined = self.w_ih.forward(input) + from_prev_hidden
        new_hidden = self.activation.forward(combined)
        output = self.w_ho.forward(new_hidden)
        return output, new_hidden


    def init_hidden(self, batch_size=1):
        return Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)




