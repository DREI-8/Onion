import torch
import onion as on
import numpy as np

data1 = np.array([ [[1, 1], [1, 1], [1, 1]], [[2,2],[2,2],[2,2]]], dtype=np.float32)
tensor1 = on.Tensor(data1)

data2 = np.array([ [[1, 1,1], [1, 1,1]], [[2,2,2],[2,2,2]] ], dtype=np.float32)
tensor2 = on.Tensor(data2)

tensor = tensor1@tensor2
data = (tensor.numpy)
print(data)