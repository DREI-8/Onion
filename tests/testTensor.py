import unittest
import numpy as np
from onion import Tensor, is_cuda_available

class TestTensor(unittest.TestCase):
    
    def setUp(self):
        """Prepare the test environment."""
        self.data = np.array([[[1, 2], [3, 4], [5, 6]], 
                             [[7, 8], [9, 10], [11, 12]]], dtype=np.float32)
        self.tensor = Tensor(self.data)

        self.data_transpose = np.array([[ 1.0028, -0.9893, 0.5809],
                                       [-0.1669, 0.7299, 0.4942]], dtype=np.float32)

        self.data_max_min = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    def test_tensor_creation(self):
        """Testing tensor creation."""
        tensor = Tensor(self.data)
        self.assertEqual(tensor.ndim, 3)
        self.assertEqual(tensor.size, 12)

    def test_get_item(self):
        """Testing access to tensor elements."""
        self.assertEqual(self.tensor.get_item([0, 0, 1]), 2)
        self.assertEqual(self.tensor.get_item([1, 2, 1]), 12)

    def test_reshape(self):
        """Testing the reshape method."""
        new_shape = [4, 3]
        reshaped_tensor = self.tensor.reshape(new_shape)
        self.assertEqual(reshaped_tensor.ndim, 2)

    def test_transpose(self):
        """Testing the transpose method."""
        tensor_transpose = Tensor(self.data_transpose)
        transposed_tensor = tensor_transpose.transpose()

        self.assertEqual(tensor_transpose.get_item([0, 0, 0]), transposed_tensor.get_item([0, 0, 0]))
        self.assertEqual(tensor_transpose.get_item([0, 1, 0]), transposed_tensor.get_item([1, 0, 0]))

    def test_reduction_operations(self):
        """Testing max, min, sum and mean methods."""
        tensor_max_min = Tensor(self.data_max_min)
        
        # Test max
        max_value = tensor_max_min.max(axis=-1, keepdims=True)
        self.assertEqual(max_value.get_item([0, 0]), 6) # max value from the entire tensor
        
        # Test min
        min_value = tensor_max_min.min(axis=-1, keepdims=False)
        self.assertEqual(min_value.get_item([0]), 1) # min value from the entire tensor
        
        # Test sum
        sum_value = tensor_max_min.sum(axis=0, keepdims=True)
        self.assertEqual(sum_value.get_item([0, 0]), 5) # [0, 0] + [1, 0] = 1 + 4
        self.assertEqual(sum_value.get_item([0, 2]), 9) # [0, 2] + [1, 2] = 2 + 5
        
        # Test mean
        mean_value = tensor_max_min.mean(axis=1, keepdims=False)
        self.assertEqual(mean_value.get_item([0]), 2.0) # mean of [1, 2, 3] = (1+2+3)/3
        self.assertEqual(mean_value.get_item([1]), 5.0) # mean of [4, 5, 6] = (4+5+6)/3

    def test_arithmetic_operations(self):
        """Testing arithmetic operations."""
        tensor2 = Tensor(np.array([[[1, 1], [1, 1], [1, 1]], 
                                  [[1, 1], [1, 1], [1, 1]]], dtype=np.float32))
        
        # Addition & Subtraction
        result = self.tensor + tensor2 - tensor2
        self.assertEqual(result.get_item([1, 2, 1]), 12)
        
        # Subtraction & Multiplication
        result = self.tensor - tensor2 * tensor2
        self.assertEqual(result.get_item([1, 2, 1]), 11)
        
        # Multiplication & Addition
        result = self.tensor * tensor2 + tensor2
        self.assertEqual(result.get_item([1, 2, 1]), 13)

    @unittest.skipIf(not is_cuda_available(), "CUDA is not available")
    def test_cuda_operations(self):
        """Testing CUDA operations (skip if CUDA is not available)."""
        cuda_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        cuda_tensor = Tensor(cuda_data)

        cuda_tensor = cuda_tensor.to("cuda")
        self.assertTrue(cuda_tensor.is_cuda())

        cuda_result_add = cuda_tensor + cuda_tensor
        cuda_result_sub = cuda_tensor - cuda_tensor
        
        # Get results back to CPU
        cpu_result_add = cuda_result_add.to("cpu")
        self.assertFalse(cpu_result_add.is_cuda())
        self.assertEqual(cpu_result_add.get_item([1, 2]), 12.0)
        
        cpu_result_sub = cuda_result_sub.to("cpu")
        self.assertFalse(cpu_result_sub.is_cuda())
        self.assertEqual(cpu_result_sub.get_item([1, 2]), 0.0)

if __name__ == '__main__':
    unittest.main()