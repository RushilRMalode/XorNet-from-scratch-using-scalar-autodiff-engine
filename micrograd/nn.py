from micrograd.engine import Value
import random

class neuron:
    #activation (weights*inputs + bias) 
    def __init__(self,n_inputs,operation='ReLU'):  
        self.weights = [Value(random.uniform(-1,1)) for _ in range(n_inputs)]
        if operation == 'ReLU':
            self.activation = Value.relu
        elif operation == 'tanh':
            self.activation = Value.tanh
        elif operation == 'sigmoid':
            self.activation = Value.sigmoid
        else:
            raise ValueError('Activation not implemented')
        
        self.bias = Value(random.uniform(-1,1))
    
    def __call__(self,x):
        x = [xi if isinstance(xi,Value) else Value(xi) for xi in x]
        out = (sum((xi*wi for xi,wi in zip(x,self.weights)),self.bias))
        out = self.activation(out)
        return out
    def parameters(self):
        return self.weights + [self.bias]

    
class layer:
    def __init__(self,width,n_inputs,operation='ReLU'):
        self.neurons = [neuron(n_inputs,operation) for _ in range(width)]
    def __call__(self,x):
        x = [xi if isinstance(xi,Value) else Value(xi) for xi in x]
        out = [n(x) for n in self.neurons]
        return out
    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params
    
class MLP:
    def __init__(self,n_layers:int,n_inputs:int,arch:list,activations:list):
        self.layers = []
        assert len(arch) == n_layers
        assert len(activations) == n_layers
        temp = n_inputs
        for num in range(n_layers):
            self.layers.append(layer(arch[num],temp,activations[num]))
            temp=arch[num]
    def __call__(self,x):
        x = [xi if isinstance(xi,Value) else Value(xi) for xi in x]
        out = x
        for _layer in self.layers:
            out = _layer(out)
        return out    
    def parameters(self):
        params = []
        for _layer in self.layers:
            params.extend(_layer.parameters())
        return params
    def zero_grads(self):
        for p in self.parameters():
            p.zero_grad()