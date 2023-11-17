from sklearn.metrics import pairwise_distances
import numpy as np
from utils.loss import TripletLoss
import torch

loss_function = TripletLoss()

a = np.array([[1.1, 2.2,], [2.2, 3.3], [3.3, 4.4]])
b = np.array([[2.2, 3.3], [3.3, 4.4], [4.4, 5.5]])


print(a)

#0.09656584