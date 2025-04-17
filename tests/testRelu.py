import unittest
import numpy as np
import torch
from onion import Tensor
from onion.nn import relu as ReLU

class TestReLU(unittest.TestCase):
    def test_relu_positive_negative_values(self):
        """Test que ReLU fonctionne correctement avec des valeurs positives et négatives."""
        # Données de test avec valeurs positives et négatives
        test_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        
        # Configure Onion
        onion_tensor = Tensor(test_data.copy())
        onion_result = ReLU(onion_tensor)
        
        # Configure PyTorch
        torch_tensor = torch.tensor(test_data.copy())
        torch_result = torch.nn.functional.relu(torch_tensor)

        self.assertTrue(np.allclose(
            onion_result.numpy(),
            torch_result.detach().numpy(),
            atol=1e-6
        ), "Les résultats ReLU diffèrent pour des valeurs positives et négatives")
    
    def test_relu_only_negative_values(self):
        """Test que ReLU fonctionne correctement avec uniquement des valeurs négatives."""
        test_data = np.array([-5.0, -4.0, -3.0, -2.0, -1.0], dtype=np.float32)
        
        onion_tensor = Tensor(test_data.copy())
        onion_result = ReLU(onion_tensor)
        
        torch_tensor = torch.tensor(test_data.copy())
        torch_result = torch.nn.functional.relu(torch_tensor)

        self.assertTrue(np.allclose(
            onion_result.numpy(),
            torch_result.detach().numpy(),
            atol=1e-6
        ), "Les résultats ReLU diffèrent pour des valeurs uniquement négatives")
    
    def test_relu_only_positive_values(self):
        """Test que ReLU fonctionne correctement avec uniquement des valeurs positives."""
        test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        
        onion_tensor = Tensor(test_data.copy())
        onion_result = ReLU(onion_tensor)
        
        torch_tensor = torch.tensor(test_data.copy())
        torch_result = torch.nn.functional.relu(torch_tensor)

        self.assertTrue(np.allclose(
            onion_result.numpy(),
            torch_result.detach().numpy(),
            atol=1e-6
        ), "Les résultats ReLU diffèrent pour des valeurs uniquement positives")
    
    def test_relu_with_zeros(self):
        """Test que ReLU fonctionne correctement avec des valeurs nulles."""
        test_data = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        onion_tensor = Tensor(test_data.copy())
        onion_result = ReLU(onion_tensor)
        
        
        
        torch_tensor = torch.tensor(test_data.copy())
        torch_result = torch.nn.functional.relu(torch_tensor)

        self.assertTrue(np.allclose(
            onion_result.numpy(),
            torch_result.detach().numpy(),
            atol=1e-6
        ), "Les résultats ReLU diffèrent pour des valeurs nulles")
    
    def test_relu_multidimensional(self):
        """Test que ReLU fonctionne correctement avec des tenseurs multidimensionnels."""
        test_data = np.array([[-2.0, 1.0], [0.0, -3.0], [4.0, -5.0]], dtype=np.float32)
        
        onion_tensor = Tensor(test_data.copy())
        onion_result = ReLU(onion_tensor)
        
        torch_tensor = torch.tensor(test_data.copy())
        torch_result = torch.nn.functional.relu(torch_tensor)

        self.assertTrue(np.allclose(
            onion_result.numpy(),
            torch_result.detach().numpy(),
            atol=1e-6
        ), "Les résultats ReLU diffèrent pour des tenseurs multidimensionnels")
    
    def test_relu_large_tensor(self):
        """Test que ReLU fonctionne correctement avec des tenseurs de grande taille."""
        # Création d'un grand tenseur avec des valeurs aléatoires
        np.random.seed(42)  # Pour la reproductibilité
        test_data = np.random.randn(1000, 1000).astype(np.float32)
        
        onion_tensor = Tensor(test_data.copy())
        onion_result = ReLU(onion_tensor)
        
        torch_tensor = torch.tensor(test_data.copy())
        torch_result = torch.nn.functional.relu(torch_tensor)

        self.assertTrue(np.allclose(
            onion_result.numpy(),
            torch_result.detach().numpy(),
            atol=1e-6
        ), "Les résultats ReLU diffèrent pour des tenseurs de grande taille")

if __name__ == '__main__':
    unittest.main()