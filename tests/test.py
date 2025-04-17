import unittest
import numpy as np
import torch
from onion import Tensor, is_cuda_available


data_3d = np.array([[[1, 2], [3, 4], [5, 6]], 
                        [[7, 8], [9, 10], [11, 12]]], dtype=np.float32)
onion_tensor = Tensor(data_3d)
print(onion_tensor)

axis = -1
keepdims = False

result = onion_tensor.mean(axis, keepdims)
print(result)
                    
