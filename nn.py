from engine import Value
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
            return ValueError('Activation not implemented')
        
        self.bias = Value(random.uniform(-1,1))
    
    def __call__(self,x):
        x = [xi if isinstance(xi,Value) else Value(xi) for xi in x]
        out = (sum((xi*wi for xi,wi in zip(x,self.weights)),self.bias))
        out = self.activation(out)
        return out
    def parameters(self):
        return self.weights + [self.bias]
    
class Layer:
    def __init__(self,width,n_inputs,operation='ReLU'):
        self.neurons = [neuron(n_inputs,operation) for _ in range(width)]
    def __call__(self,x):
        x = [xi if isinstance(xi,Value) else Value(xi) for xi in x]
        out = [n(x) for n in self.neurons]
        return out
    def parameters(self):
        return [n.parameters() for n in self.neurons]