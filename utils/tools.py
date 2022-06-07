import os
import shutil


# Clear files function
def removing(current_path):
    for file in os.listdir(current_path):
        current_path_file = os.path.join(current_path, file)
        if os.path.isdir(current_path_file):
            shutil.rmtree(current_path_file)
        elif os.path.isfile(current_path_file):
            os.remove(current_path_file)


class Tools:
    def __init__(self):
        pass

    # Clear caches in calculating ascension point
    def clear_cache(self, filename="*", time_limit="*"):
        import calculate_denglinPoint.config.DenglinConsts as param1

        if filename == "*":
            current_path = param1.VIEWSHED_PATH
            removing(current_path)

        elif filename != "*" and time_limit == "*":

            current_path = os.path.join(param1.VIEWSHED_PATH,
                                        str(filename))
            # 删除filename下所有步行时间的结果
            removing(current_path)

        elif filename != "*" and time_limit != "*":

            current_path = os.path.join(param1.VIEWSHED_PATH,
                                        os.path.join(str(filename),
                                                     str(time_limit)))
            # 删除某个步行时间的所有结果
            removing(current_path)

    # 清除结果文件，需要指定清除可达域计算结果还是登临点计算结果
    def clear_result(self, mode=0, filename="*", time_limit="*"):
        import calculate_denglinPoint.config.DenglinConsts as param1
        import Up_to_Domain.config.UpToDomainConsts as param2

        def clear_denglin(filename="*", time_limit="*"):

            # 删除./static/result/denglin下所有文件
            if filename == "*":
                current_path = param1.ASCENSIONLIST_SAVEPATH
                removing(current_path)

            # 删除./static/result/denglin/filename下所有文件

            elif filename != "*" and time_limit == "*":
                current_path = os.path.join(param1.ASCENSIONLIST_SAVEPATH, str(filename))
                removing(current_path)

            # 删除./static/result/denglin/filename/time_limit下所有文件

            elif filename != "*" and time_limit != "*":

                current_path = os.path.join(param1.ASCENSIONLIST_SAVEPATH,
                                            os.path.join(str(filename),
                                                         str(time_limit)))
                removing(current_path)

        def clear_reachableArea(filename="*", time_limit="*"):
            # 暂时只清除可达域计算结果，保留源数据的展示结果
            # 清除reachableImg目录下所有村落的子文件夹
            if filename == "*":
                current_path = param2.REACHABLEIMG_PATH
                removing(current_path)

            # 清除reachableImg/filename下所有文件
            elif filename != "*" and time_limit == "*":
                current_path = os.path.join(param2.REACHABLEIMG_PATH, str(filename))
                removing(current_path)

        #     默认清除可达域和登临点所有计算结果
        if mode == 0:
            #         清除登临点文件
            clear_denglin()
            #         清除可达域文件
            clear_reachableArea()

        # 当mode不为0时，清除登临点和可达域目录下的filename和time_limit是不同的，不会冲突
        elif mode == 1:
            # 仅清除登临点
            clear_denglin(filename, time_limit)
        elif mode == 2:
            # 仅清除可达域
            clear_reachableArea(filename, time_limit)
