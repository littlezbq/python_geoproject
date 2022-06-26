import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ascensionpoint_generate.interfaces import interface_visionpoint
from Up_to_Domain.interfaces import up_to_domain

# filepath = r"D:\projects\webGisProject_backEnd\static\result\village_edge\SX3_005_QMCEdge.png"
# data = np.asarray(Image.open(filepath).convert('L'))
# data[data == 38] = 255
# res = Image.fromarray(data)
# res.save("village_edge.png")




if __name__ == "__main__":
    # 跑可达域的程序
    # up_to_domain.interface_uptodomain(dem_path=r"D:\projects\webGisProject_backEnd\static\datasets\SX3_005_QMC.tif",
    #                                   remote_path=r"D:\projects\webGisProject_backEnd\static\datasets\SX3_005_QMC.png",
    #                                   timelimit=15,
    #                                   demAxis=8.5)

    # 跑登临点的程序
    # interface_visionpoint.interface_cal_ascensionpoint(dem_path=r"D:\projects\webGisProject_backEnd\static\datasets\SX3_005_QMC.tif",
    #                       denglin_num=3,
    #                       time_limit=15,
    #                       divideSpace=30,
    #                       demAxis=8.5
    #                       )
