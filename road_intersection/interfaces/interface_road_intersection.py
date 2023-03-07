from PIL import Image
from cv2 import imwrite
from road_intersection.road_extraction import detect_road
from config.params import road_intersection_path,road_path
from road_intersection.road_intersection_extration import detect_road_intersection
import os


def interface_get_road_intersection(remote_path):
    '''
    Get Road Intersection from a remote sensing or UAV image(RGB type)
    :param remote_path: Path of input remote sensing image
    :return: save path of road graph and intersection
    '''
    image_name = os.path.basename(remote_path).split(".")[0]
    # road_graph_name = os.path.join(road_path, image_name + "_road_graph.png")
    # road_intersection_name = os.path.join(road_intersection_path,image_name + "_road_intersections.png")


    road_graph_name = remote_path
    road_intersection_name = os.path.join(r"G:\projects\webGisProject_backEnd20220903\static\result\roadIntersection\road_intersections",image_name + "_road_intersections.png")

    # image = Image.open(remote_path)

    # Get road graph and save it
    # if os.path.exists(road_graph_name) is False:
    #     res_road = detect_road.get_road(image)
    #     res_road.save(road_graph_name)

    # Get road intersection and save it
    if os.path.exists(road_intersection_name) is False:
        res_road_intersection = detect_road_intersection.get_road_intersections(road_graph_name)
        # res_road_intersection.save(road_intersection_name)
        imwrite(road_intersection_name,res_road_intersection)

    return road_graph_name, "/api/" + road_intersection_name


if __name__ == "__main__":
    img_path = "../datas/roads_alone"
    road_list = os.listdir(img_path)
    [interface_get_road_intersection(os.path.join(img_path,road)) for road in road_list]
