import unittest, sys
import numpy as np

from src.toy_dl.tensor.tensor import Tensor

class Test(unittest.TestCase): 

    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.test_data = [1, 2, 3, 4, 5, 10]


    def test_initial(self):
        x = Tensor(self.test_data)
        y = np.array(self.test_data)

        self.assertEqual(x.__str__(), y.__str__())



    def test_add(self):
        x = Tensor(self.test_data)
        y = np.array(self.test_data)

        z = x + y
        
        self.assertEqual(z.__str__(), np.array([2,4,6, 8,10, 20]).__str__())


    def test_backprop_a(self): 
        x = Tensor(self.test_data, autograd=True)
        y = Tensor(self.test_data, autograd=True)
        z = Tensor(self.test_data, autograd=True)


        xx = x + y 
        yy = y + z
        zz = xx + yy

        try: 
            zz.backward(Tensor(np.array([1, 1, 1, 1, 1, 1])))
        except Exception as E: 
            sys.stdout.write(E.__str__())
            self.fail(E.__str__())

    



    # auto gradient descent.
    def test_backprop_b(self): 
        x = Tensor(self.test_data, autograd=True)
        y = Tensor(self.test_data, autograd=True)

        # adding these two add the two to its list of creators
        z = x + y 


        z.backward(Tensor(np.array([1,1,1,1,1,1])))

        self.assertEqual(x.grad, y.grad)

        test_creators = []
        test_creators.append(x)
        test_creators.append(y)


        self.assertEqual(z.creators, test_creators)
        
    
    # negation gradient
    def test_negation(self): 
        x = Tensor(self.test_data, autograd=True)
        y = Tensor(self.test_data, autograd=True)
        z = Tensor(self.test_data, autograd=True)
        

        a = x + (-y)
        b = (-y) + z
        c = a + b
        
        c.backward(Tensor(np.array([1, 1, 1, 1, 1, 1])))

        self.assertEqual(y.grad.data.__str__(), np.array([-2, -2, -2, -2, -2, -2]).__str__())


    # sum test
    def test_sum(self):
        x = Tensor(np.array([[1, 2, 3], 
                             [4, 5, 6]]))

        # shrink into a smaller dimension
        value = x.sum(0)
        self.assertEqual(value.__str__(), np.array([5, 7, 9]).__str__())

        # shrink into a smaller dimension
        value = x.sum(1)
        self.assertEqual(value.__str__(), np.array([6, 15]).__str__())


        # expand into a larger dimension (copying values into its larger dimension)
        y = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        value2 = y.expand(dim=0, copies=2)
        
        self.assertEqual(value2.__str__(), np.array([[[1, 2, 3], [4, 5, 6]],[[1, 2, 3], [4, 5, 6]]]).__str__())


    def test_train(self):
        import numpy as np
        np.random.seed(0)
        data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
        target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

        w = list()
        w.append(Tensor(np.random.rand(2, 3), autograd=True))
        w.append(Tensor(np.random.rand(3, 1), autograd=True))

        for i in range(10):
            pred = data.mm(w[0]).mm(w[1]) # prediction layer
            loss = ((pred - target)*(pred - target)).sum(0) # comparison
            loss.backward(Tensor(np.ones_like(loss.data))) # backprop
            for w_ in w: 
                w_.data -= w_.grad.data * 0.1
                w_.grad.data *= 0

        self.assertGreater(0.7, loss.data)

            

