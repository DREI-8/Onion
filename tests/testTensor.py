import unittest
import numpy as np
import torch
from onion import Tensor, is_cuda_available

class TestTensor(unittest.TestCase):
    
    def setUp(self):
        """Prepare test data for different scenarios."""
        # Basic 3D data
        self.data_3d = np.array([[[1, 2], [3, 4], [5, 6]], 
                                [[7, 8], [9, 10], [11, 12]]], dtype=np.float32)
        
        # 2D data for reductions
        self.data_2d = np.array([[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0]], dtype=np.float32)
        
        # Edge case: 1D data
        self.data_1d = np.array([-1.5, 2.3, 0.0, 4.7], dtype=np.float32)
        
        # Non-contiguous data (transposed)
        self.data_non_contig = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32).T
        
        # Zero-initialized data
        self.data_zeros = np.zeros((2, 3), dtype=np.float32)

    def _compare_with_torch(self, onion_result, np_input, axis, keepdims, op_name):
        """Helper to compare results with PyTorch implementation."""
        # Convert to PyTorch tensor
        torch_tensor = torch.from_numpy(np_input)
        
        # PyTorch operation
        if op_name == 'max':
            torch_result = torch_tensor.max(dim=axis, keepdim=keepdims)[0]
        elif op_name == 'min':
            torch_result = torch_tensor.min(dim=axis, keepdim=keepdims)[0]
        elif op_name == 'sum':
            torch_result = torch_tensor.sum(dim=axis, keepdim=keepdims)
        elif op_name == 'mean':
            torch_result = torch_tensor.mean(dim=axis, keepdim=keepdims)
        
        # Convert results to numpy
        onion_np = onion_result.numpy()
        torch_np = torch_result.numpy()
        
        # Compare shapes and values
        self.assertEqual(onion_np.shape, torch_np.shape,
                         f"Shape mismatch for {op_name} (axis={axis}, keepdims={keepdims})")
        self.assertTrue(np.allclose(onion_np, torch_np, atol=1e-6),
                        f"Value mismatch for {op_name} (axis={axis}, keepdims={keepdims})")

    def _test_operation(self, np_data, device, op_name, axes_to_test):
        """Generic test for reduction operations across devices."""
        # Create our tensor and move to device
        onion_tensor = Tensor(np_data).to(device)
        
        for axis in axes_to_test:
            for keepdims in [True, False]:
                with self.subTest(device=device, op=op_name, axis=axis, keepdims=keepdims):
                    # Perform operation
                    if op_name == 'max':
                        result = onion_tensor.max(axis, keepdims)
                    elif op_name == 'min':
                        result = onion_tensor.min(axis, keepdims)
                    elif op_name == 'sum':
                        result = onion_tensor.sum(axis, keepdims)
                    elif op_name == 'mean':
                        result = onion_tensor.mean(axis, keepdims)
                    
                    # Move result to CPU for comparison
                    if result.is_cuda():
                        result = result.to("cpu")
                    
                    # Compare with PyTorch
                    self._compare_with_torch(result, np_data, axis, keepdims, op_name)
    def _compare_with_nan(self, a, b, atol=1e-6):
        """Compare two arrays, allowing for NaN values."""
        a_nan = np.isnan(a)
        b_nan = np.isnan(b)
        if not np.array_equal(a_nan, b_nan):
            return False
        
        return np.allclose(
            a[~a_nan],  
            b[~b_nan],  
            atol=atol
        )

    def test_reductions_cpu(self):
        """Test reduction operations on CPU."""
        test_cases = [
            (self.data_2d, ['max', 'min', 'sum', 'mean'], [-1, 0, 1]),
            (self.data_1d, ['max', 'min', 'sum', 'mean'], [-1, 0]),
            (self.data_non_contig, ['max', 'sum'], [0, 1]),
            (self.data_zeros, ['min', 'mean'], [0])
        ]
        
        for data, ops, axes in test_cases:
            for op in ops:
                self._test_operation(data, "cpu", op, axes)

    @unittest.skipIf(not is_cuda_available(), "CUDA not available")
    def test_reductions_cuda(self):
        """Test reduction operations on CUDA."""
        test_cases = [
            (self.data_2d, ['max', 'min', 'sum', 'mean'], [-1, 0, 1]),
            (self.data_1d, ['max', 'min', 'sum', 'mean'], [-1, 0]),
            (self.data_non_contig, ['max', 'sum'], [0, 1]),
            (self.data_zeros, ['min', 'mean'], [0])
        ]
        
        for data, ops, axes in test_cases:
            for op in ops:
                self._test_operation(data, "cuda", op, axes)

    def test_arithmetic_consistency(self):
        """Test arithmetic operations across devices against PyTorch."""
        ops = [
            ('add', lambda a, b: a + b),
            ('sub', lambda a, b: a - b),
            ('mul', lambda a, b: a * b),
            ('div', lambda a, b: a / b)
        ]
        
        test_data = [
            (self.data_2d, self.data_2d),
            (self.data_1d, self.data_1d),
            (self.data_non_contig, self.data_non_contig)
        ]
        
        for device in ["cpu", "cuda"] if is_cuda_available() else ["cpu"]:
            if device == "cuda" and not is_cuda_available():
                continue
                
            for a_np, b_np in test_data:
                # Create tensors
                a_onion = Tensor(a_np).to(device)
                b_onion = Tensor(b_np).to(device)
                
                # Create PyTorch tensors
                a_torch = torch.from_numpy(a_np).to(device)
                b_torch = torch.from_numpy(b_np).to(device)
                
                for op_name, op_func in ops:
                    with self.subTest(device=device, op=op_name):
                        # Onion operation
                        result_onion = op_func(a_onion, b_onion)
                        if result_onion.is_cuda():
                            result_onion = result_onion.to("cpu")
                        
                        # PyTorch operation
                        result_torch = op_func(a_torch, b_torch).cpu().numpy()
                        
                        # Comparison
                        self.assertTrue(self._compare_with_nan(result_onion.numpy(), result_torch, atol=1e-6))

    def test_scalar_arithmetic_consistency(self):
        """Test arithmetic operations between tensors and scalars against PyTorch."""
        # tensor op scalar operations
        ops = [
            ('add', lambda t, s: t + s),
            ('sub', lambda t, s: t - s),
            ('mul', lambda t, s: t * s),
            ('div', lambda t, s: t / s)
        ]
        
        # scalar op tensor operations
        r_ops = [
            ('radd', lambda s, t: s + t),
            ('rsub', lambda s, t: s - t),
            ('rmul', lambda s, t: s * t),
            # ('rdiv', lambda s, t: s / t)  # Not implemented yet in Onion
        ]
        
        # Test scalars
        scalars = [2.0, 0.5, -1.0]
        
        # Tensors Data
        test_data = [
            self.data_2d,
            self.data_1d,
            self.data_non_contig
        ]
        
        for device in ["cpu", "cuda"] if is_cuda_available() else ["cpu"]:
            if device == "cuda" and not is_cuda_available():
                continue
                
            for data_np in test_data:
                # Create tensors
                t_onion = Tensor(data_np).to(device)
                t_torch = torch.from_numpy(data_np).to(device)
                
                # Test tensor op scalar
                for op_name, op_func in ops:
                    for scalar in scalars:
                        with self.subTest(device=device, op=op_name, scalar=scalar):
                            # Onion operation
                            result_onion = op_func(t_onion, scalar)
                            if result_onion.is_cuda():
                                result_onion = result_onion.to("cpu")
                            
                            # PyTorch operation
                            result_torch = op_func(t_torch, scalar).cpu().numpy()
                            
                            # Comparison
                            self.assertTrue(np.allclose(result_onion.numpy(), result_torch, atol=1e-6),
                                        f"Mismatch for {op_name} with scalar={scalar}")
                
                # Test scalar op tensor
                for op_name, op_func in r_ops:
                    for scalar in scalars:
                        with self.subTest(device=device, op=op_name, scalar=scalar):
                            # Onion operation
                            result_onion = op_func(scalar, t_onion)
                            if result_onion.is_cuda():
                                result_onion = result_onion.to("cpu")
                            
                            # PyTorch operation
                            result_torch = op_func(scalar, t_torch).cpu().numpy()
                            
                            # Comparison
                            self.assertTrue(np.allclose(result_onion.numpy(), result_torch, atol=1e-6),
                                        f"Mismatch for {op_name} with scalar={scalar}")
                        
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Invalid axis
        t = Tensor(self.data_2d)
        with self.assertRaises(RuntimeError):
            t.max(axis=3)
        
        # Empty tensor
        empty_t = Tensor(np.array([], dtype=np.float32))
        self.assertEqual(empty_t.sum().numpy().item(), 0.0)

        # All NaN values
        nan_data = np.full((2, 2), np.nan, dtype=np.float32)
        nan_t = Tensor(nan_data)
        self.assertTrue(np.isnan(nan_t.sum().numpy().item()))

    @unittest.skipIf(not is_cuda_available(), "CUDA not available")
    def test_device_conversion(self):
        """Test device conversion mechanics."""
        # CPU -> CUDA -> CPU
        t = Tensor(self.data_2d)
        t_gpu = t.to("cuda")
        self.assertTrue(t_gpu.is_cuda())
        t_cpu = t_gpu.to("cpu")
        self.assertFalse(t_cpu.is_cuda())
        self.assertTrue(np.allclose(t.numpy(), t_cpu.numpy()))

        # Operations across devices should fail
        t_gpu2 = Tensor(self.data_2d).to("cuda")
        with self.assertRaises(RuntimeError):
            t + t_gpu2

if __name__ == '__main__':
    unittest.main()