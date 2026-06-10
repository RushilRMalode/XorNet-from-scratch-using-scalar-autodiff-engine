import unittest
import math
from micrograd.engine import Value

class TestValueActivations(unittest.TestCase):
    
    def setUp(self):
        # A small epsilon for finite difference numerical gradient checking
        self.h = 1e-6
        self.eps = 1e-4

    def test_tanh_forward_and_backward(self):
        x_val = 0.5
        
        # 1. Forward Pass Check
        a = Value(x_val)
        c = a.tanh()
        
        expected_forward = (math.exp(x_val) - math.exp(-x_val)) / (math.exp(x_val) + math.exp(-x_val))
        self.assertAlmostEqual(c.data, expected_forward, places=6)
        
        # 2. Backward Pass Check
        c.backward()
        
        # Numerical gradient approximation: (f(x+h) - f(x-h)) / 2h
        c_plus = Value(x_val + self.h).tanh()
        c_minus = Value(x_val - self.h).tanh()
        numerical_grad = (c_plus.data - c_minus.data) / (2 * self.h)
        
        self.assertAlmostEqual(a.grad, numerical_grad, delta=self.eps)

    def test_sigmoid_forward_and_backward(self):
        x_val = -0.2
        
        # 1. Forward Pass Check
        a = Value(x_val)
        c = a.sigmoid()
        
        expected_forward = 1 / (1 + math.exp(-x_val))
        self.assertAlmostEqual(c.data, expected_forward, places=6)
        
        # 2. Backward Pass Check
        c.backward()
        
        # Numerical gradient approximation
        c_plus = Value(x_val + self.h).sigmoid()
        c_minus = Value(x_val - self.h).sigmoid()
        numerical_grad = (c_plus.data - c_minus.data) / (2 * self.h)
        
        self.assertAlmostEqual(a.grad, numerical_grad, delta=self.eps)

    def test_relu_positive(self):
        x_val = 1.5 # Positive input
        a = Value(x_val)
        c = a.relu()
        
        # Forward
        self.assertEqual(c.data, 1.5)
        
        # Backward
        c.backward()
        self.assertEqual(a.grad, 1.0)

    def test_relu_negative(self):
        x_val = -1.5 # Negative input
        a = Value(x_val)
        c = a.relu()
        
        # Forward
        self.assertEqual(c.data, 0.0)
        
        # Backward
        c.backward()
        self.assertEqual(a.grad, 0.0)

    def test_pow_forward_and_backward(self):
        x_val = 3.0
        power = 3
        
        # 1. Forward Pass Check
        a = Value(x_val)
        c = a ** power
        
        self.assertEqual(c.data, 27.0)
        
        # 2. Backward Pass Check
        c.backward()
        
        # Analytical power rule derivative: n * x^(n-1) -> 3 * 3^2 = 27
        self.assertEqual(a.grad, 27.0)

    def test_gradient_accumulation_multi_use(self):
        # Crucial test to make sure += works and gradients don't overwrite
        a = Value(2.0)
        b = a.relu()
        c = a.tanh()
        d = b + c # 'a' splits into two branches here
        
        d.backward()
        
        # Analytical check: d(d)/da = d(relu)/da + d(tanh)/da
        # For x=2.0: d(relu) = 1.0; d(tanh) = 1 - tanh(2.0)^2
        t = (math.exp(2) - math.exp(-2)) / (math.exp(2) + math.exp(-2))
        expected_grad = 1.0 + (1.0 - t**2)
        
        self.assertAlmostEqual(a.grad, expected_grad, places=6)

if __name__ == '__main__':
    unittest.main()