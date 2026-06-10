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
        self._op = _op # operation
    def zero_grad(self):
        self.grad = 0

    def __repr__(self):
        # How to display the value object
        return f'Value : {self.data}, grad : {self.grad}'

    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self,other),'+')
        def _backward():
            self.grad += 1*out.grad; other.grad += 1*out.grad
        out._backward = _backward
        return out
    
    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self,other),'*')
        def _backward():
            self.grad += other.data*out.grad; other.grad += self.data*out.grad
        out._backward = _backward
        return out
    
    def __pow__(self,other):
        assert isinstance(other,(int,float))
        out = Value(self.data**other,(self,),_op=f'**{other}')
        def _backward():
            self.grad+=other*(self.data**(other-1))*out.grad
        out._backward=_backward
        return out
    def tanh(self):
        def val_tanh(x):
            return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
        t=val_tanh(self.data)
        out = Value(t,(self,),_op='tanh')#need to pass (self,) not self as children since it needs to be iterable, othewise define __iter__
        def _backward():
            # g = tanh
            # g'(x) = 1-(g(x))^2
            self.grad+=(1-t**2)*out.grad
        out._backward = _backward
        return out
    def sigmoid(self):
        def val_sigmoid(x):
            return 1/(1+math.exp(-x))
        t=val_sigmoid(self.data)
        out = Value(t,(self,),_op='sigmoid')
        def _backward():
            # g = sigmoid
            # g'(x) = g(x)(1-g(x))
            self.grad+=(t)*(1-t)*out.grad
        out._backward=_backward
        return out
    
    def relu(self):
        out = Value((self.data>0)*self.data,(self,),_op='ReLU')
        out._backward = lambda:setattr(self,"grad",self.grad + (self.data>0)*out.grad) #self.grad+=(self.data>0)*out.grad, this is just a single liner
        return out

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
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