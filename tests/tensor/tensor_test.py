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



    # auto gradient descent.
    def test_backprop(self): 
        x = Tensor(self.test_data)
        y = Tensor(self.test_data)

        # adding these two add the two to its list of creators
        z = x + y 


        z.backward(Tensor(np.array([1,1,1,1,1,1])))

        print(z.creators)



