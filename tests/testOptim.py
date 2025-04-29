import unittest
import numpy as np
import torch
from onion import Tensor
from onion.optim import Adam

class TestOptimizers(unittest.TestCase):
    def test_adam_single_step(self):
        """Test that Adam optimizer performs correct parameter updates after a single step."""
        param_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        grad_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        # Configure Onion
        onion_param = Tensor(param_data.copy(), requires_grad=True)
        onion_param.set_grad(Tensor(grad_data.copy()))
        onion_adam = Adam([onion_param], lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8)
        
        # Configure PyTorch
        torch_param = torch.tensor(param_data.copy(), requires_grad=True)
        torch_param.grad = torch.tensor(grad_data.copy())
        torch_adam = torch.optim.Adam([torch_param], lr=0.01, betas=(0.9, 0.999), eps=1e-8)

        onion_adam.step()
        torch_adam.step()

        self.assertTrue(np.allclose(
            onion_param.numpy(),
            torch_param.detach().numpy(),
            atol=1e-6
        ), "The parameter updates differ after a single step")
    
    def test_adam_multiple_steps(self):
        """Test that Adam optimizer performs correct updates over multiple steps."""
        param_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        # Configure Onion
        onion_param = Tensor(param_data.copy(), requires_grad=True)
        onion_adam = Adam([onion_param], lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
        
        # Configure PyTorch
        torch_param = torch.tensor(param_data.copy(), requires_grad=True)
        torch_adam = torch.optim.Adam([torch_param], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
        
        # Simulate multiple steps
        for i in range(5):
            grad_data = np.array([0.1, 0.2, 0.3], dtype=np.float32) * (i + 1)

            # Update gradients
            onion_param.set_grad(Tensor(grad_data.copy()))
            torch_param.grad = torch.tensor(grad_data.copy())

            onion_adam.step()
            torch_adam.step()
            
            self.assertTrue(np.allclose(
                onion_param.numpy(),
                torch_param.detach().numpy(),
                atol=1e-6
            ), f"The parameter updates differ after step {i+1}")
            
            # Put gradients to zero for the next step
            onion_adam.zero_grad()
            torch_adam.zero_grad()

if __name__ == '__main__':
    unittest.main()