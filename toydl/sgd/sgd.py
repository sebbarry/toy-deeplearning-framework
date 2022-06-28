"""
Here we will pass in the weights of the network. 
The parameters that we loop through are what we adjust during the process of SGD.
"""

class SGD(object):

    def __init__(self, parameters, alpha=0.1):
        self.parameters = parameters
        self.alpha = alpha

    # zero out the data in the gradient values of the  parameters 
    def zero(self):
        for p in self.parameters:
            p.grad.data *= 0

    # apply the weight adjustment function here.
    # multiplying w/ alpha value to add a small amount
    # of noise to the adjustment during gd
    def step(self, zero=True):
        for p in self.parameters: 
            p.data -= p.grad.data * self.alpha # same as weights_0_1 -= alpha * delta_layer_1
            if(zero):
                p.grad.data *= 0






