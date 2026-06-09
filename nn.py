from engine import Value

class neuron:
    #activation (weights*inputs + bias) 
    def __init__(self,n_inputs,operation='ReLU'):  
        self.weights = [Value(0)]*n_inputs
        if operation == 'ReLU':
            self.activation = Value.relu
        elif operation == 'tanh':
            self.activation = Value.tanh
        elif operation == 'sigmoid':
            self.activation = Value.sigmoid
        else:
            return NameError
        
        self.bias = Value(0)
    
    def __call__(self,x):
        out = Value(x*self.weights.data+self.bias.data,_children=(self.weights,self.bias),_op= '+')
        out = self.activation(out)
        return out
    def parameters(self):
        return self.weights
    
