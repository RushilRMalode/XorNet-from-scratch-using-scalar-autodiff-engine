import unittest
from micrograd.nn import neuron, layer, MLP
from micrograd.engine import Value

class TestNN(unittest.TestCase):
    def test_handling_neuron(self):
        x = [-1]
        my_neuron = neuron(n_inputs=1)
        out = my_neuron(x)
        self.assertIsInstance(out,Value)
        print(out.data)
    def test_handling_layer(self):
        x=[-1]
        my_layer=layer(n_inputs=1,width=5)
        out = my_layer(x)
        self.assertIsInstance(out[0],Value)
        print(out)
    def test_handling_MLP(self):
        x=[-1]
        my_MLP = MLP(n_layers=3,n_inputs=1,arch=[1,2,1],activations=['ReLU','ReLU','ReLU'])
        out = my_MLP(x)
        self.assertIsInstance(out[0],Value)
        print(out)
if __name__ == '__main__':
    unittest.main()        
    