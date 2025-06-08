import deepxde as dde
from typing import List

def make_fnn(layer_sizes: List[int], activation="tanh"):
    return dde.maps.FNN(layer_sizes, activation, "Glorot normal")