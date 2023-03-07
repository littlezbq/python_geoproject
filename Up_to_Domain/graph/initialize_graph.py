import itertools
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# matplotlib.use('agg')
import networkx as nx
# from config import params
from Up_to_Domain.cal_accessarea.point import Point


class DEM_Graph():
    def __init__(self, dem_data, demAxis):
        self.dem = dem_data
        self.demAxis = demAxis  # Default axis
        self.im_height, self.im_width = dem_data.shape[0], dem_data.shape[1]  # 行，列
        self.G = self.create_graph()

    def get_sptial_distance(self, currentPoint, tmpPoint):
        """
        估算计算两格网的空间距离
        :param currentPoint:
        :param tmpPoint:
        :return:
        """
        space_distance = np.sqrt(
            np.square(np.abs(currentPoint.x - tmpPoint.x) * self.demAxis) +
            np.square(np.abs(currentPoint.y - tmpPoint.y) * self.demAxis)
        )

        sptial_distance = np.sqrt(
            np.square(space_distance) +
            np.square(
                np.abs(self.dem[int(currentPoint.x)][int(currentPoint.y)] - self.dem[int(tmpPoint.x)][int(tmpPoint.y)]))
        )

        return sptial_distance

    def cal_weight(self, i, j, i_side, j_side):
        """
        生成dem图结构两节点的长度
        :param i:
        :param j:
        :param i_side:
        :param j_side:
        :return:
        """
        former_point = Point(i, j)
        latter_point = Point(i_side, j_side)
        tmp_point_sptial_distance = self.get_sptial_distance(former_point, latter_point)
        sum = tmp_point_sptial_distance

        return sum

    def create_graph(self):
        """
        根据DEM格网八邻域结构创建图结构，节点为各格网，边为格网之间的欧氏距离。根据坡度大小，对dem进行剪枝操作，大于45°的坡度格网之间的边将
        被移除
        :return:
        """
        start = time.time()
        self.G = nx.Graph()

        i_list = [i for i in range(self.im_height)]
        j_list = [j for j in range(self.im_width)]

        points = list(itertools.product(i_list, j_list))

        for point in points:
            # 取出当前节点添加进图
            i, j = point[0],point[1]
            side_nodes = []
            if self.dem[i][j] != -9999:
                node_index = i * self.im_width + j
                self.G.add_node(node_index)
                #         添加边,默认添加八邻域
                side_nodes = [
                    [i - 1, j - 1],  # up-left
                    [i - 1, j],  # up
                    [i - 1, j + 1],  # up-right
                    [i, j - 1],  # left
                    [i, j + 1],  # right
                    [i + 1, j - 1],  # down-left
                    [i + 1, j],  # down
                    [i + 1, j + 1],  # down-right
                ]

            # 添加该节点的八邻域
            if side_nodes != [] and len(side_nodes) > 0:
                for sidenode in side_nodes:
                    i_side, j_side = sidenode[0], sidenode[1]
                    if 0 <= i_side < self.im_height and 0 <= j_side < self.im_width:
                        if self.dem[i_side][j_side] != -9999:
                            # Consider distance between nodes as weight
                            weight_i_j = self.cal_weight(i, j, i_side, j_side)
                            node_side_index = i_side * self.im_width + j_side

                            # 计算两个节点之间的坡度正切值
                            threshold = abs(self.dem[i][j] - self.dem[i_side][j_side]) / self.demAxis
                            # 封装邻域节点
                            self.G.add_node(node_side_index)
                            # 只添加坡度小于45度的
                            if 0 <= threshold < 1:
                                self.G.add_edge(node_index, node_side_index, weight=weight_i_j)

        end = time.time()
        print("Creating DEM_Graph in {:.2f} seconds".format(end - start))

        return self.G

    def save_graph(self):
        # save_path = params.GRAPH_PATH
        nx.write_graphml_lxml(self.G, "demnetwork.graphml")

    def shortest_path(self, src):
        """
        Using dijkstra method to cal_shorest path
        :param src: 
        :param dst: 
        :return: 
        """
        #     求两点最短路径
        if self.G is None:
            print("Error occured")
            exit(-1)
        # shortest_list = nx.astar_path(self.G, src, dst)
        shortest_list = nx.shortest_path(self.G, source=src, method="dijkstra", weight="weight")

        return shortest_list


if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    from commonutils.tools import Tools
    # dem_data = np.array(Image.open(r'G:\projects\datas\ChengGuiData\dem\Hancheng_s_new.tif'))
    dem_data, _, _, _ = Tools.read_dem(r'G:\projects\datas\ChengGuiData\dem\Hancheng_s_new_s.tif')

    dem_graph = DEM_Graph(dem_data,30)
    # dem_graph.draw_graph()

    # dem_graph.save_graph()

    srcNode = 40 * dem_data.shape[1] + 60
    shortest_list = dem_graph.shortest_path(srcNode)

    dstNode = 65 * dem_data.shape[1] + 109

    paths = shortest_list[dstNode]

    plt.figure()
    plt.imshow(dem_data)
    for path in paths:
        path_x = path // dem_data.shape[1]
        path_y = path - dem_data.shape[1] * path_x

        if path == srcNode:
            plt.scatter(path_y, path_x, c='blue', s=10)
        else:
            plt.scatter(path_y, path_x, c='red', s=10)

    plt.show()
