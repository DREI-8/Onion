import onion as o
import numpy as np

print("=== Test de la bibliothèque Onion ===")
print(f"Fonctions et classes disponibles: {dir(o)}")
print(f"Test de base: {o.test()}")  # Print "Onion is working!"

# Create a numpy array  
data = np.array([[[1, 2], [3, 4], [5, 6]], 
                 [[7, 8], [9, 10], [11, 12]]], dtype=np.float32)
print(f"\nDonnées numpy originales:\n{data}")

# Create a tensor from the numpy array
tensor = o.Tensor(data)
print(f"\nTenseur créé avec succès!")
print(f"Nombre de dimensions: {tensor.ndim}")
print(f"Taille totale: {tensor.size}")

# get_item method test
print("\nTest d'accès aux éléments:")
print(f"tensor[0,0,1] = {tensor.get_item([0,0,1])}")
print(f"tensor[1,2,1] = {tensor.get_item([1,2,1])}")

# reshape method test
try:
    new_shape = [4, 3]
    reshaped_tensor = tensor.reshape(new_shape)
    print(f"\nTenseur reshapé avec succès à la forme {new_shape}")
    print(f"Nouvelle dimension: {reshaped_tensor.ndim}")
    print(f"tensor[0,0,1] = {tensor.get_item([0,0,1])}")
    print(f"tensor[1,2,1] = {tensor.get_item([1,2,1])}")
except Exception as e:
    print(f"\nErreur lors du reshape: {e}")
    
try:
    # addition
    tensor2 = o.Tensor(np.array([[[1, 1], [1, 1], [1, 1]], 
                                  [[1, 1], [1, 1], [1, 1]]], dtype=np.float32))
    result = tensor + tensor2 - tensor2
    print("\nAddition réussie!")
    print(f"Résultat de l'addition:\n{result.get_item([1,2,1])}")
except Exception as e:
    print(f"\nErreur lors de l'addition: {e}")
    
try:
    # soustraction
    result = tensor - tensor2 * tensor2
    print("\nSoustraction réussie!")
    print(f"Résultat de la soustraction:\n{result.get_item([1,2,1])}")
except Exception as e:
    print(f"\nErreur lors de la soustraction: {e}")
    
try:
    # multiplication
    result = tensor * tensor2 + tensor2
    print("\nMultiplication réussie!")
    print(f"Résultat de la multiplication:\n{result.get_item([1,2,1])}")
except Exception as e:
    print(f"\nErreur lors de la multiplication: {e}")

# CUDA Tests
print("\n=== Tests CUDA ===")
print(f"CUDA disponible: {o.is_cuda_available()}")

try:
    # Create tensor for CUDA operations
    cuda_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    cuda_tensor = o.Tensor(cuda_data)
    
    # Move to CUDA if available
    if o.is_cuda_available():
        cuda_tensor = cuda_tensor.to("cuda")
        print(f"Tenseur déplacé sur CUDA: {cuda_tensor.is_cuda()}")
        
        # GPU operations - Addition
        cuda_result_add = cuda_tensor + cuda_tensor
        print("Addition sur GPU réussie!")
        
        # GPU operations - Subtraction
        cuda_result_sub = cuda_tensor - cuda_tensor
        print("Soustraction sur GPU réussie!")
        
        # Move results back to CPU
        cpu_result_add = cuda_result_add.to("cpu")
        print(f"Résultat d'addition ramené sur CPU: {not cpu_result_add.is_cuda()}")
        print(f"Valeur à [1,2]: {cpu_result_add.get_item([1,2])}")
        
        cpu_result_sub = cuda_result_sub.to("cpu")
        print(f"Résultat de soustraction ramené sur CPU: {not cpu_result_sub.is_cuda()}")
        print(f"Valeur à [1,2]: {cpu_result_sub.get_item([1,2])}")
        print(f"Nombre de dimensions: {cpu_result_sub.ndim}")
    else:
        print("CUDA n'est pas disponible, tests GPU ignorés.")
except Exception as e:
    print(f"Erreur lors des tests CUDA: {e}")

print("\n=== Fin des tests ===")