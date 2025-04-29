import unittest
import numpy as np
from onion import Tensor, is_cuda_available

class TestTensorMatMul(unittest.TestCase):
    
    def setUp(self):
        # Données de test pour les cas 2D
        self.a_2d = np.random.randn(3, 4).astype(np.float32)
        self.b_2d = np.random.randn(4, 5).astype(np.float32)
        
        # Données de test pour les cas 3D
        self.a_3d = np.random.randn(2, 3, 4).astype(np.float32)
        self.b_3d = np.random.randn(2, 4, 5).astype(np.float32)
        
        # Données pour le broadcasting
        self.b_3d_broadcast = np.random.randn(4, 5).astype(np.float32)

    def compare_with_numpy(self, onion_result, np_input1, np_input2):
        """Utilitaire pour comparer avec les résultats de numpy"""
        np_result = np.matmul(np_input1, np_input2)
        np.testing.assert_allclose(onion_result.numpy(), np_result, rtol=1e-5)

    def test_matmul_2d(self):
        # Test CPU
        tensor_a = Tensor(self.a_2d)
        tensor_b = Tensor(self.b_2d)
        result = tensor_a.matmul(tensor_b)
        self.compare_with_numpy(result, self.a_2d, self.b_2d)
        
        # Test shape
        self.assertEqual(result.shape, (3, 5))

    def test_matmul_3d(self):
        # Test CPU
        tensor_a = Tensor(self.a_3d)
        tensor_b = Tensor(self.b_3d)
        result = tensor_a.matmul(tensor_b)
        self.compare_with_numpy(result, self.a_3d, self.b_3d)
        
        # Test shape
        self.assertEqual(result.shape, (2, 3, 5))

    def test_matmul_mixed_dims(self):
        # Cas 1: 3D x 2D
        tensor_a = Tensor(self.a_3d)
        tensor_b = Tensor(self.b_2d)
        result = tensor_a.matmul(tensor_b)
        self.compare_with_numpy(result, self.a_3d, self.b_2d)
        self.assertEqual(result.shape, (2, 3, 5))
        
        # Cas 2: 2D x 3D
        tensor_a = Tensor(self.a_2d)
        tensor_b = Tensor(self.b_3d)
        result = tensor_a.matmul(tensor_b)
        self.compare_with_numpy(result, self.a_2d, self.b_3d)
        self.assertEqual(result.shape, (2, 3, 5))

    def test_matmul_errors(self):
        # Dimensions incompatibles
        a = Tensor(np.random.randn(3, 4))
        b = Tensor(np.random.randn(5, 3))
        with self.assertRaises(RuntimeError):
            a.matmul(b)
            
        # Batch size mismatch
        a = Tensor(np.random.randn(2, 3, 4))
        b = Tensor(np.random.randn(3, 4, 5))
        with self.assertRaises(RuntimeError):
            a.matmul(b)
            
        # Device mismatch
        if is_cuda_available():
            a_cuda = a.to("cuda")
            with self.assertRaises(RuntimeError):
                a_cuda.matmul(b)

    @unittest.skipIf(not is_cuda_available(), "CUDA non disponible")
    def test_matmul_cuda(self):
        # Test 2D sur GPU
        tensor_a = Tensor(self.a_2d).to("cuda")
        tensor_b = Tensor(self.b_2d).to("cuda")
        result = tensor_a.matmul(tensor_b).to("cpu")
        self.compare_with_numpy(result, self.a_2d, self.b_2d)
        
        # Test 3D sur GPU
        tensor_a = Tensor(self.a_3d).to("cuda")
        tensor_b = Tensor(self.b_3d).to("cuda")
        result = tensor_a.matmul(tensor_b).to("cpu")
        self.compare_with_numpy(result, self.a_3d, self.b_3d)
        
        # Test broadcasting sur GPU
        tensor_a = Tensor(self.a_3d).to("cuda")
        tensor_b = Tensor(self.b_3d_broadcast).to("cuda")
        result = tensor_a.matmul(tensor_b).to("cpu")
        self.compare_with_numpy(result, self.a_3d, self.b_3d_broadcast)

    @unittest.skipIf(not is_cuda_available(), "CUDA non disponible")
    def test_matmul_gradient(self):
        # Test de la propagation du device
        tensor_a = Tensor(self.a_2d).to("cuda")
        tensor_b = Tensor(self.b_2d)
        with self.assertRaises(RuntimeError):
            tensor_a.matmul(tensor_b)

if __name__ == '__main__':
    unittest.main()