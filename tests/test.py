import onion as o
import numpy as np

print("=== Test de la bibliothèque Onion ===")
print(f"Fonctions et classes disponibles: {dir(o)}")
print(f"Test de base: {o.test()}")  # Doit afficher "Hello, Onion!"

# Création d'un tenseur à partir d'un numpy array
data = np.array([[[1, 2], [3, 4], [5, 6]], 
                 [[7, 8], [9, 10], [11, 12]]], dtype=np.float32)
print(f"\nDonnées numpy originales:\n{data}")

# Création du tenseur
tensor = o.Tensor(data)
print(f"\nTenseur créé avec succès!")
print(f"Nombre de dimensions: {tensor.ndim}")
print(f"Taille totale: {tensor.size}")

# Test de get_item
print("\nTest d'accès aux éléments:")
print(f"tensor[0,0,0] = {tensor.get_item([0,0,1])}")
print(f"tensor[1,2,1] = {tensor.get_item([1,2,1])}")

# Test de reshape si disponible
try:
    new_shape = [4, 3]
    reshaped_tensor = tensor.reshape(new_shape)
    print(f"\nTenseur reshapé avec succès à la forme {new_shape}")
    print(f"Nouvelle dimension: {reshaped_tensor.ndim}")
    print(f"tensor[0,0,0] = {tensor.get_item([0,0,1])}")
    print(f"tensor[1,2,1] = {tensor.get_item([1,2,1])}")
except Exception as e:
    print(f"\nErreur lors du reshape: {e}")

# Test des opérations si disponibles
try:
    # Crée un second tenseur pour les opérations
    data2 = np.ones_like(data) * 2
    tensor2 = o.Tensor(data2)
    
    # Teste l'addition si disponible
    if hasattr(o, 'add'):
        result = o.add(tensor, tensor2)
        print(f"\nRésultat de l'addition: Taille={result.size}, Dimensions={result.ndim}")
        print(f"Premier élément: {result.get_item([0,0,0])}")
except Exception as e:
    print(f"\nErreur lors des opérations: {e}")

print("\n=== Fin des tests ===")