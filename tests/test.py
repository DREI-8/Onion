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

# Transpose method test 
try:
    print("\n=== Test de la méthode transpose ===")
    # Create a numpy array for testing transpose
    data_transpose = np.array([[ 1.0028, -0.9893,  0.5809],
        [-0.1669,  0.7299,  0.4942]], dtype=np.float32)
    print(f"\nDonnées numpy originales pour transpose:\n{data_transpose}")
    
    # Create a tensor from the numpy array
    tensor_transpose = o.Tensor(data_transpose)
    print(f"\nTenseur créé avec succès pour transpose!")
    print(f"Shape avant transpose: {tensor_transpose}")
    
    # Transpose the tensor
    transposed_tensor = tensor_transpose.transpose()
    print(f"\nTenseur transposé avec succès!")
    print(f"Shape après transpose: {transposed_tensor}")
    
    # Print some values to check the transpose
    print(f"tensor_transpose[0,0,0] = {tensor_transpose.get_item([0,0,0])}")
    print(f"transposed_tensor[0,0,0] = {transposed_tensor.get_item([0,0,0])}")
    print(f"tensor_transpose[0,1,0] = {tensor_transpose.get_item([0,1,0])}")
    print(f"transposed_tensor[1,0,0] = {transposed_tensor.get_item([1,0,0])}")

except Exception as e:
    print(f"\nErreur lors du transpose: {e}")
    
# Test max and min methods
try:
    data_max_min = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    tensor_max_min = o.Tensor(data_max_min)
    print(f"\nTenseur créé avec succès pour max/min!")
    print(f"Tensor: {tensor_max_min}")
    
    # Test max
    max_value = tensor_max_min.max(axis=-1, keepdims=True)
    print(f"\nValeur maximale le long de l'axe: {max_value}")
    
    # Test min
    min_value = tensor_max_min.min(axis=-1, keepdims=False)
    print(f"\nValeur minimale le long de l'axe: {min_value}")
except Exception as e:
    print(f"\nErreur lors du max/min: {e}")
    
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