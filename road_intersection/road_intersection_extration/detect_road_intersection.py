from road_intersection.road_intersection_extration.tensor_voting import TensorVoting

def get_road_intersections(image_path):
    '''
    Get intersections of an input binary road graph image
    :param image: path of input binary road graph
    :return: binary image of intersections
    '''

    model = TensorVoting(image_path)

    intersections_road,intersections = model.run()

    cv2.imwrite(r"G:\projects\webGisProject_backEnd20220903\road_intersection\space_syntax\test_data\intersection.png",intersections)

    return intersections


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    image_path = r"G:\projects\webGisProject_backEnd20220903\road_intersection\space_syntax\test_data\10078750_15.png"
    res = get_road_intersections(image_path)

    # plt.figure()
    # plt.imshow(res)
    # plt.show()
    # cv2.namedWindow("image")
    # cv2.imshow("image", res)
    # cv2.waitKey()
    # cv2.destroyAllWindows()