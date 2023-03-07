import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd
from rasterio.features import shapes


def read_shp(shp_file):
    """

    :param shpe_file:
    :return:
    """
    # 读取shp文件
    data = gpd.read_file(shp_file)
    print(data.head())
    return data


def img2shp(img_file, shp_file):
    """

    :param img_file:
    :param shp_file:
    :return:
    """
    with rasterio.open(img_file) as src:
        image = src.read(1)
        transform = src.transform

    # 对分割图进行矢量化
    results = (
        {'properties': {'val': v}, 'geometry': s}
        for i, (s, v) in enumerate(
        shapes(image, mask=None, transform=transform)
    )
    )

    # 将矢量数据转换成GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(list(results))

    # 保存为shp文件
    gdf.to_file(shp_file)


def show_shp(shp_file):
    """

    :param shp_file:
    :return:
    """
    # 读取shp文件
    data = gpd.read_file(shp_file)

    # 绘制地图
    fig, ax = plt.subplots(figsize=(10, 10))
    data.plot(ax=ax)

    # 显示地图
    plt.show()

if __name__ == "__main__":
    img_file = './test_data/intersection.png'
    shp_file = './test_data/intersection.shp'
    # img2shp(img_file, shp_file)
    read_shp(shp_file)