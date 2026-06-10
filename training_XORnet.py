from micrograd.nn import MLP
from micrograd.engine import Value
import random

N_data = 500
data_x = [random.randint(0,1) for _ in range(N_data)]
data_y = [random.randint(0,1) for _ in range(N_data)]
target = [Value(xi^yi) for xi,yi in zip(data_x,data_y)]

xor_net = MLP(n_layers=2,n_inputs=2,arch=[2,1],activations=['tanh','sigmoid'])

epochs = 100
loss_history = []
lr=0.01
for epoch in range(epochs):
    #evaluate loss
    xor_net.zero_grads()
    loss=Value(0)
    for n in range(len(data_x)):
        x = [data_x[n],data_y[n]]
        t = target[n]
        out = xor_net(x)
        loss += (t-out[0])**2
    
    #w_new = w_old - lr * dloss/dw
    loss.backward()
    for p in xor_net.parameters():
        p.data-=lr*p.grad
N_test = 100
test_x = [random.randint(0,1) for _ in range(N_test)]
test_y = [random.randint(0,1) for _ in range(N_test)]
errors=0
eps = 1e-3
for xi,yi in zip(test_x,test_y):
    out = xor_net([xi,yi])
    if abs(out[0].data - (xi^yi))>0.5:
        errors+=1

print(f'Accuracy {(N_test - errors)/N_test}')