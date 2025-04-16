import unittest
import numpy as np
from onion import Tensor, Linear, is_cuda_available

class TestLinear(unittest.TestCase):
    
    def setUp(self):
        """Prepare the test environment."""
        # Create sample input data
        self.input_data = np.array([[1.0, 2.0, 3.0], 
                                   [4.0, 5.0, 6.0]], dtype=np.float32)
        self.input_tensor = Tensor(self.input_data)
        
        # Create Linear layers for testing
        self.in_features = 3
        self.out_features = 2
        #self.linear_with_bias = Linear(self.in_features, self.out_features, True)
        self.linear_without_bias = Linear(self.in_features, self.out_features, False)

    def test_linear_creation(self):
        """Testing linear layer creation."""
        # Check dimensions of weights
        self.assertEqual(self.linear_without_bias.weights.shape[0],self.in_features )
        self.assertEqual(self.linear_without_bias.weights.shape[1],self.out_features )
        
        # Check dimensions of bias when bias is used
        self.assertEqual(self.linear_without_bias.bias.shape[0], self.out_features)
        
        # Check bias is None when not used
        self.assertEqual(self.linear_without_bias.bias.size, 0)

    def test_linear_forward(self):
        """Testing the forward pass through the linear layer."""
        # Test with bias
        # output_with_bias = self.linear_with_bias.apply(self.input_tensor)
        # self.assertEqual(output_with_bias.shape[0], self.input_tensor.shape[0])
        # self.assertEqual(output_with_bias.shape[1], self.out_features)
        
        # Test without bias
        output_without_bias = self.linear_without_bias.apply(self.input_tensor)
        self.assertEqual(output_without_bias.shape[0], self.input_tensor.shape[0])
        self.assertEqual(output_without_bias.shape[1], self.out_features)
        
        # Calculate expected output manually (for a simple case)
        # For this test, we'll set the weights and bias to known values
        test_weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        test_bias = np.array([0.1, 0.2], dtype=np.float32)
        
        # Create a new linear layer with known weights and bias
        linear_test = Linear(self.in_features, self.out_features, False)
        linear_test.weights = Tensor(test_weights)
        #linear_test.bias = Tensor(test_bias)
        
        output_test = linear_test.apply(self.input_tensor)
        
        # Expected output for first batch element: [1.0, 2.0, 3.0] @ [0.1, 0.2, 0.3]^T + 0.1 = 1.4
        # and [1.0, 2.0, 3.0] @ [0.4, 0.5, 0.6]^T + 0.2 = 3.2
        expected_first_row = np.array([1.4, 3.2], dtype=np.float32)
        
        # Check if results are close (considering floating-point precision)
        self.assertTrue(np.abs(output_test.get_item([0, 0]) - expected_first_row[0]) < 1e-5)
        self.assertTrue(np.abs(output_test.get_item([0, 1]) - expected_first_row[1]) < 1e-5)

    def test_device_movement(self):
        """Testing movement between CPU and CUDA."""
        # Create a linear layer on CPU
        linear_cpu = Linear(self.in_features, self.out_features, False, "cpu")
        self.assertFalse(linear_cpu.weights.is_cuda())
        self.assertFalse(linear_cpu.bias.is_cuda())
        
        # Skip CUDA tests if not available
        if not is_cuda_available():
            return
        
        # Move to CUDA
        linear_cpu.to("cuda")
        self.assertTrue(linear_cpu.weights.is_cuda())
        self.assertTrue(linear_cpu.bias.is_cuda())
        
        # Create a linear layer directly on CUDA
        linear_cuda = Linear(self.in_features, self.out_features, False, "cuda")
        self.assertTrue(linear_cuda.weights.is_cuda())
        self.assertTrue(linear_cuda.bias.is_cuda())
        
        # Move input tensor to CUDA
        cuda_input = self.input_tensor.to("cuda")
        
        # Test forward pass on CUDA
        cuda_output = linear_cuda.apply(cuda_input)
        self.assertTrue(cuda_output.is_cuda())
        
        # Move back to CPU
        linear_cuda.to("cpu")
        self.assertFalse(linear_cuda.weights.is_cuda())
        self.assertFalse(linear_cuda.bias.is_cuda())

    @unittest.skipIf(not is_cuda_available(), "CUDA is not available")
    def test_cuda_operations(self):
        """Testing CUDA operations (skip if CUDA is not available)."""
        # Create a linear layer on CUDA
        linear_cuda = Linear(self.in_features, self.out_features, False, "cuda")
        
        # Move input tensor to CUDA
        cuda_input = self.input_tensor.to("cuda")
        
        # Test forward pass
        cuda_output = linear_cuda.apply(cuda_input)
        
        # Move back to CPU for verification
        cpu_output = cuda_output.to("cpu")
        
        # Check output dimensions
        self.assertEqual(cpu_output.shape[0], self.input_tensor.shape[0])
        self.assertEqual(cpu_output.shape[1], self.out_features)
        
        # Compare results with CPU computation
        linear_cpu = Linear(self.in_features, self.out_features, True, "cpu")
        # Set the weights and bias to be the same as the CUDA version
        linear_cpu.weights = linear_cuda.weights.to("cpu")
        linear_cpu.bias = linear_cuda.bias.to("cpu")
        
        cpu_output_direct = linear_cpu.apply(self.input_tensor)
        
        # Check that results are equal (within floating-point precision)
        for i in range(self.input_tensor.shape[0]):
            for j in range(self.out_features):
                self.assertTrue(np.abs(cpu_output.get_item([i, j]) - cpu_output_direct.get_item([i, j])) < 1e-5)



if __name__ == '__main__':
    unittest.main()