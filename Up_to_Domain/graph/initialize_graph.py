import networkx as nx
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_graph(m,n):
    G = nx.grid_2d_graph(m,n)
    return G

if __name__ == "__main__":
    dem_path = r"D:\projects\webGisProject_backEnd\static\datasets\SX3_005_QMC.tif"
    dem_data = np.asarray(Image.open(dem_path))
    m, n = dem_data.shape[0], dem_data.shape[1]

    G = create_graph(m,n)
    plt.figure()
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()
