class Value:
    '''
    the class to store the created objects
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
        return out
    
    def __mul__(self,other):
        out = Value(self.data * other.data, (self,other),'*')
        def _backward():
            self.grad += 1*out.grad; other.grad += 1*out.grad

        return out
            
    def backward(self):
        #going over all variables in topographical order to propagate the gradient backward
        topo =[]
        visited = set()
        return