from sklearn.metrics import pairwise_distances
import numpy as np
from utils.loss import TripletLoss
import torch

loss_function = TripletLoss()

a = np.array([[1.1, 2.2,], [2.2, 3.3], [3.3, 4.4]])
b = np.array([[2.2, 3.3], [3.3, 4.4], [4.4, 5.5]])

loss = loss_function(torch.tensor(a), torch.tensor(b))  

# computing the loss in hand
# we get image loss (cost_im in loss.py)
# [[0, 0.0915, 0.0855]; [0.1015, 0, 0.0977]; [0.0989, 0.1005, 0]]
# and recipe loss (cost_s in loss.py)
# [[0, 0.854, 0.078]; [0.1077, 0, 0.0967], [0.1061, 0.1015, 0]]

print(a)

#0.09656584