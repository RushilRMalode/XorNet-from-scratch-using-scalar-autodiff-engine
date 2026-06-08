import math

class Value:
    '''
    a way to store the objects
    '''
    def __init__(self,data,_children=(),_op=''):
        # Main object data
        self.data = data
        self.grad = 0
        # internal  
        self._backward = lambda : None #backward prop function based on operation
        self._prev = set(_children) # set of variables making this variable up via operations
        self.op = _op # operation

    def __repr__(self):
        # How to print a value class object
        print(f'Value : {self.data}, grad : {self.grad}')

    def __add__(self,other):
        out = Value(self.data + other.data, (self,other),'+')
        def _backward():
            self.grad += 1*out.grad; other.grad += 1*out.grad
        out._backward = _backward
        return out
    
    def __mul__(self,other):
        out = Value(self.data * other.data, (self,other),'*')
        def _backward():
            self.grad += 1*out.grad; other.grad += 1*out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        def val_tanh(x):
            return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
        
        out = Value(val_tanh(self.data),self,_op='tanh')
        def _backward():
            # g = tanh
            # g'(x) = 1-(g(x))^2
            self.grad+=(1-val_tanh(self.data)**2)*out.grad
        out._backward = _backward
    def sigmoid(self):
        def val_sigmoid(x):
            return 1/(1+math.exp(-x))
        out = Value(val_sigmoid(self.data),self,_op='sigmoid')
        def _backward():
            # g = sigmoid
            # g'(x) = g(x)(1-g(x))
            self.grad+=(val_sigmoid(self.data))*(1-val_sigmoid(self.data))*out.grad
        out._backward=_backward
    def relu(self):
        out = Value((self.data>0)*self.data,self,_op='relu')
        out._backward = lambda:setattr(self,"grad",self.grad + (self.data>0)*out.grad) #self.grad+=(self.data>0)*out.grad, this is just a single liner
    
    def backward(self):
        #going over all variables in topographical order to propagate the gradient backward                
        topo=[]
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad=1.0
        for v in reversed(topo):
            v._backward()  
        return