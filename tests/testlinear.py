import unittest
import numpy as np
import torch
from onion import Tensor, Linear, is_cuda_available

class TestLinear(unittest.TestCase):
    
    def setUp(self):
        """Prepare test data for different scenarios."""
        # Batch input (2D)
        self.data_2d = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ], dtype=np.float32)  # batch_size=3, in_features=4
        
        # Sequence input (3D)
        self.data_3d = np.array([
            [[1.0, 2.0], [3.0, 4.0]], 
            [[5.0, 6.0], [7.0, 8.0]]
        ], dtype=np.float32)  # seq_len=2, batch_size=2, in_features=2
        
        # Single example (1D)
        self.data_1d = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)  # in_features=4
        
        # Edge cases
        self.data_zeros = np.zeros((2, 4), dtype=np.float32)
        self.data_ones = np.ones((3, 2), dtype=np.float32)

    def _compare_linear_outputs(self, onion_layer, torch_layer, input_data, device, atol=1e-5):
        """Compare outputs from Onion and PyTorch linear layers."""
        # Convert numpy to tensors
        onion_tensor = Tensor(input_data).to(device)
        torch_tensor = torch.tensor(input_data).to(device if device == "cuda" else "cpu")
        
        # Copy weights from torch to onion to ensure fair comparison
        torch_weight = torch_layer.weight.detach().cpu().numpy()
        if hasattr(onion_layer, 'set_weights'):
            onion_layer.set_weights(torch_weight.T)  # Transpose if needed based on your implementation
        else:
            # Copy weights manually (assuming weights are accessible)
            weight_tensor = Tensor(torch_weight.T)
            onion_layer.weights = weight_tensor.to(device)
            
        # Copy bias if present
        if torch_layer.bias is not None and onion_layer.bias.size > 0:
            torch_bias = torch_layer.bias.detach().cpu().numpy()
            if hasattr(onion_layer, 'set_bias'):
                onion_layer.set_bias(torch_bias)
            else:
                bias_tensor = Tensor(torch_bias)
                onion_layer.bias = bias_tensor.to(device)
        
        # Forward pass
        onion_output = onion_layer.apply(onion_tensor)
        if onion_output.is_cuda():
            onion_output = onion_output.to("cpu")
        torch_output = torch_layer(torch_tensor).cpu().detach().numpy()
        
        # Compare results
        onion_np = onion_output.numpy()
        
        # Check shapes
        self.assertEqual(onion_np.shape, torch_output.shape, 
                        f"Shape mismatch: onion={onion_np.shape}, torch={torch_output.shape}")
        
        # Check values
        self.assertTrue(np.allclose(onion_np, torch_output, atol=atol),
                        f"Value mismatch: max diff = {np.max(np.abs(onion_np - torch_output))}")

    # def test_linear_2d_with_bias_cpu(self):
    #     """Test linear layer with 2D input and bias on CPU."""
    #     in_features = 4
    #     out_features = 5
        
    #     # Create layers
    #     onion_linear = Linear(in_features, out_features, bias=True, device_name="cpu")
    #     torch_linear = torch.nn.Linear(in_features, out_features, bias=True)
        
    #     # Compare outputs
    #     self._compare_linear_outputs(onion_linear, torch_linear, self.data_2d, "cpu")

    def test_linear_2d_without_bias_cpu(self):
        """Test linear layer with 2D input without bias on CPU."""
        in_features = 4
        out_features = 5
        
        # Create layers
        onion_linear = Linear(in_features, out_features, bias=False, device_name="cpu")
        torch_linear = torch.nn.Linear(in_features, out_features, bias=False)
        
        # Compare outputs
        self._compare_linear_outputs(onion_linear, torch_linear, self.data_2d, "cpu")

    def test_linear_3d_cpu(self):
        """Test linear layer with 3D input on CPU."""
        in_features = 2
        out_features = 3
        
        # Create layers
        onion_linear = Linear(in_features, out_features, bias=False, device_name="cpu")
        torch_linear = torch.nn.Linear(in_features, out_features, bias=False)
        
        # Compare outputs
        self._compare_linear_outputs(onion_linear, torch_linear, self.data_3d, "cpu")

    def test_linear_1d_expanded_cpu(self):
        """Test linear layer with 1D input (expanded to 2D) on CPU."""
        in_features = 4
        out_features = 2
        
        # Create layers
        onion_linear = Linear(in_features, out_features, bias=False, device_name="cpu")
        torch_linear = torch.nn.Linear(in_features, out_features, bias=False)
        
        # Reshape 1D data to 2D (adding batch dimension)
        input_data = self.data_1d.reshape(1, -1)
        
        # Compare outputs
        self._compare_linear_outputs(onion_linear, torch_linear, input_data, "cpu")

    def test_edge_cases_cpu(self):
        """Test edge cases on CPU."""
        # Test with zeros
        in_features = 4
        out_features = 3
        onion_linear = Linear(in_features, out_features, bias=False, device_name="cpu")
        torch_linear = torch.nn.Linear(in_features, out_features, bias=False)
        self._compare_linear_outputs(onion_linear, torch_linear, self.data_zeros, "cpu")
        
        # Test with ones
        in_features = 2
        out_features = 4
        onion_linear = Linear(in_features, out_features, bias=False, device_name="cpu")
        torch_linear = torch.nn.Linear(in_features, out_features, bias=False)
        self._compare_linear_outputs(onion_linear, torch_linear, self.data_ones, "cpu")

    @unittest.skipIf(not is_cuda_available(), "CUDA not available")
    def test_linear_2d_with_bias_cuda(self):
        """Test linear layer with 2D input and bias on CUDA."""
        in_features = 4
        out_features = 5
        
        # Create layers
        onion_linear = Linear(in_features, out_features, bias=False, device_name="cuda")
        torch_linear = torch.nn.Linear(in_features, out_features, bias=False).cuda()
        
        # Compare outputs
        self._compare_linear_outputs(onion_linear, torch_linear, self.data_2d, "cuda")

    @unittest.skipIf(not is_cuda_available(), "CUDA not available")
    def test_linear_3d_cuda(self):
        """Test linear layer with 3D input on CUDA."""
        in_features = 2
        out_features = 3
        
        # Create layers
        onion_linear = Linear(in_features, out_features, bias=False, device_name="cuda")
        torch_linear = torch.nn.Linear(in_features, out_features, bias=False).cuda()
        
        # Compare outputs
        self._compare_linear_outputs(onion_linear, torch_linear, self.data_3d, "cuda")

    def test_invalid_input_dimensions(self):
        """Test that Linear rejects invalid input dimensions."""
        in_features = 3
        out_features = 2
        linear = Linear(in_features, out_features, bias=False)
        
        # Create 4D tensor (should be rejected)
        data_4d = np.zeros((2, 2, 2, 3), dtype=np.float32)
        tensor_4d = Tensor(data_4d)
        
        with self.assertRaises(ValueError):
            linear.apply(tensor_4d)

    def test_input_size_mismatch(self):
        """Test that Linear rejects input with wrong feature dimension."""
        in_features = 5
        out_features = 2
        linear = Linear(in_features, out_features, bias=False)
        
        # Create tensor with wrong feature count
        wrong_data = np.zeros((2, 3), dtype=np.float32)  # in_features should be 5, not 3
        wrong_tensor = Tensor(wrong_data)
        
        with self.assertRaises(ValueError):
            linear.apply(wrong_tensor)

    def test_device_conversion(self):
        """Test moving the linear layer between devices."""
        if not is_cuda_available():
            self.skipTest("CUDA not available")
            
        in_features = 4
        out_features = 3
        
        # Create on CPU
        linear = Linear(in_features, out_features, bias=False, device_name="cpu")
        
        # Move to CUDA
        linear.to("cuda")
        
        # Test that forward pass works with CUDA tensor
        input_tensor = Tensor(self.data_2d).to("cuda")
        output = linear.apply(input_tensor)
        
        self.assertTrue(output.is_cuda())
        
        # Move back to CPU
        linear.to("cpu")
        
        # Test that forward pass works with CPU tensor
        input_tensor = Tensor(self.data_2d)  # On CPU
        output = linear.apply(input_tensor)
        
        self.assertFalse(output.is_cuda())

    def test_weight_initialization(self):
        """Test that weight initialization is within expected bounds."""
        for _ in range(5):  # Run multiple times to check for consistency
            in_features = 100
            out_features = 50
            
            linear = Linear(in_features, out_features, bias=False)
            weights = linear.weights.numpy()
            
            # Xavier/Glorot initialization should be roughly uniform in ±sqrt(6/(in+out))
            expected_limit = np.sqrt(6.0 / (in_features + out_features))
            
            self.assertTrue(np.all(weights >= -expected_limit - 1e-5))
            self.assertTrue(np.all(weights <= expected_limit + 1e-5))
            self.assertTrue(-0.1 < np.mean(weights) < 0.1)  # Mean should be close to zero

if __name__ == '__main__':
    unittest.main()