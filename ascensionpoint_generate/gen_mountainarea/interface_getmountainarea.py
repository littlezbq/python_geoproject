import numpy as np
from PIL import Image
from skimage import measure


def interface_getmountainarea(dem_path):
    """
    :param dem: ndarray type dem data of the village
    :return: the main mountain area of the dem,
    using (min_longtitude, max_latitude, max_longtitude, min_latitude) format
    """
    # Generate the border index of mountain area, (row,column) format
    dem = np.asarray(Image.open(dem_path))
    contours = measure.find_contours(dem, level=(np.nanmin(dem) + np.nanmax(dem)) / 2.0)

    # Result contour: (min_longtitude, max_latitude, max_longtitude, min_latitude), left top and right bottom
    result_contours = []
    # Each contour consists relative mountain area indexs
    for contour in contours:
        # Finding max and min of row and column
        max_row, max_column = np.max(contour, axis=0)
        min_row, min_column = np.min(contour, axis=0)

        # Add the left top and right bottom point to list
        result_contours.append([np.ceil(min_column), np.ceil(min_row), np.ceil(max_column), np.ceil(max_row)])

        #         Convert the coords to longtitude and latitude, if want to check on gdal bula bula
        '''
            tl = Tools()
            [max_longtitude, max_latitude],_ = tl.coordinateTransform(filename=dem_path, x=np.ceil(max_column), y=np.ceil(max_row))
            [min_longtitude, min_latitude],_ = tl.coordinateTransform(filename=dem_path, x=np.ceil(min_column), y=np.ceil(min_row))
            result_contours.append([min_longtitude, max_latitude, max_longtitude, min_latitude])
        '''

    return result_contours
