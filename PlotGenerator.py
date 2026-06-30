import matplotlib.pyplot as plt
import numpy as np

class PlotGenerator:
    def __init__(self):
        ...
    
    def heatmap2d(arr: np.ndarray):
        plt.imshow(arr, cmap='viridis')
        plt.colorbar()
        plt.show()