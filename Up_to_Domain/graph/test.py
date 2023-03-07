import networkx as nx
import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
#
# G = nx.Graph()
# G.add_edge(1, 2, color="red")
# G.add_edge(2, 3, color="red")
# G.add_node(3)
# G.add_node(4)
#
# # A = nx.nx_agraph.to_agraph(G)  # convert to a graphviz graph
# # A.draw("attributes.png", prog="neato")  # Draw with pygraphviz
# nx.write_graphml_lxml(G,"testnet.graphml")

# G = nx.cycle_graph(5, create_using = nx.DiGraph())
# nx.draw(G,with_labels=True)
# plt.title('有权图')
# plt.axis('on')
# plt.xticks([])
# plt.yticks([])
# plt.show()

# paths = nx.shortest_path(G,"仓井", weight="weight",method="bellman-ford")

# print(paths)


filename = r"D:\projects\webGisProject_backEnd20220903\static\datasets\SX3_005_QMC.tif"
dataset = gdal.Open(filename)
im_width = dataset.RasterXSize  # 读取图像的宽度，x方向上的像素个数
im_height = dataset.RasterYSize  # 读取图像的高度，y方向上的像素个数
im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
im_proj = dataset.GetProjection()  # 地图投影信息
im_data = dataset.ReadAsArray(0, 0, im_width, im_height).astype(np.float64)  # 将数据写成数组，对应栅格矩阵
# im_data = np.array(Image.open(filename))
del dataset  # 关闭对象，文件dataset
print(im_geotrans)
