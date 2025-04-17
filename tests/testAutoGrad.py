import onion
import torch
import numpy as np
import unittest

class TestAutoGrad(unittest.TestCase):
    def setUp(self):
        # This method will be called before each test
        pass
    
    def compare_gradients(self, onion_grad, torch_grad, name="gradients", atol=1e-6):
        """Helper method to compare gradients between onion and PyTorch"""
        onion_grad_np = onion_grad.numpy()
        torch_grad_np = torch_grad.detach().numpy()
        
        self.assertTrue(
            np.allclose(onion_grad_np, torch_grad_np, atol=atol),
            f"{name} don't match between onion and PyTorch. "
            f"Max difference: {np.max(np.abs(onion_grad_np - torch_grad_np))}"
        )

    def test_relu_autograd(self):
        # Create random data with some negative values
        data = np.random.uniform(-1, 1, (3, 4)).astype(np.float32)
        
        # Create tensors in onion and PyTorch
        x_onion = onion.Tensor(data, requires_grad=True)
        x_torch = torch.tensor(data, requires_grad=True)
        
        # Forward pass
        y_onion = onion.nn.relu(x_onion)
        y_torch = torch.nn.functional.relu(x_torch)
        
        # Verify forward pass is correct
        onion_output = y_onion.numpy()
        torch_output = y_torch.detach().numpy()
        self.assertTrue(np.allclose(onion_output, torch_output),
                      "ReLU outputs don't match")
        
        # Create upstream gradients and backward pass
        upstream_grad = np.ones_like(data)
        y_onion.backward(onion.Tensor(upstream_grad))
        y_torch.backward(torch.ones_like(y_torch))
        
        # Compare gradients
        self.compare_gradients(x_onion.grad, x_torch.grad, "ReLU gradients")
        
        # Verify ReLU gradient correctness
        expected_grad = (data > 0).astype(np.float32)
        self.assertTrue(np.allclose(x_onion.grad.numpy(), expected_grad),
                      "ReLU gradient values are incorrect")

    def test_matmul_autograd(self):
        # Create matrices for multiplication
        a_data = np.random.rand(3, 4).astype(np.float32)
        b_data = np.random.rand(4, 2).astype(np.float32)
        
        # Create tensors
        a_onion = onion.Tensor(a_data, requires_grad=True)
        b_onion = onion.Tensor(b_data, requires_grad=True)
        a_torch = torch.tensor(a_data, requires_grad=True)
        b_torch = torch.tensor(b_data, requires_grad=True)
        
        # Forward pass
        c_onion = a_onion.matmul(b_onion)
        c_torch = a_torch @ b_torch
        
        # Verify forward pass is correct
        self.assertTrue(np.allclose(c_onion.numpy(), c_torch.detach().numpy(), atol=1e-5),
                      "Matrix multiplication outputs don't match")
        
        # Create a scalar objective and backward pass
        loss_onion = c_onion.sum()
        loss_torch = c_torch.sum()
        loss_onion.backward()
        loss_torch.backward()
        
        # Compare gradients
        self.compare_gradients(a_onion.grad, a_torch.grad, "Matrix A gradients")
        self.compare_gradients(b_onion.grad, b_torch.grad, "Matrix B gradients")
        
        # Verify gradient calculations
        expected_grad_a = np.matmul(np.ones((3, 2)), b_data.T)
        self.assertTrue(np.allclose(a_onion.grad.numpy(), expected_grad_a, atol=1e-5),
                      "Matrix A gradient calculation is incorrect")
        
        expected_grad_b = np.matmul(a_data.T, np.ones((3, 2)))
        self.assertTrue(np.allclose(b_onion.grad.numpy(), expected_grad_b, atol=1e-5),
                      "Matrix B gradient calculation is incorrect")

    def test_combined_operation(self):
        # Create matrices
        a_data = np.random.uniform(-1, 1, (3, 4)).astype(np.float32)
        b_data = np.random.uniform(-1, 1, (4, 5)).astype(np.float32)
        
        # Create tensors
        a_onion = onion.Tensor(a_data, requires_grad=True)
        b_onion = onion.Tensor(b_data, requires_grad=True)
        a_torch = torch.tensor(a_data, requires_grad=True)
        b_torch = torch.tensor(b_data, requires_grad=True)
        
        # Forward pass: matmul followed by ReLU
        c_onion = a_onion.matmul(b_onion)
        d_onion = onion.nn.relu(c_onion)
        c_torch = a_torch @ b_torch
        d_torch = torch.nn.functional.relu(c_torch)
        
        # Verify forward pass
        self.assertTrue(np.allclose(d_onion.numpy(), d_torch.detach().numpy(), atol=1e-5),
                      "Combined operation outputs don't match")
        
        # Backward pass
        loss_onion = d_onion.sum()
        loss_torch = d_torch.sum()
        loss_onion.backward()
        loss_torch.backward()
        
        # Compare gradients
        self.compare_gradients(a_onion.grad, a_torch.grad, "Matrix A gradients")
        self.compare_gradients(b_onion.grad, b_torch.grad, "Matrix B gradients")


if __name__ == "__main__":
    unittest.main()
