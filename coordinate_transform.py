import os

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def transform_data(src, dst):
    # 打开源数据文件
    with rasterio.open(src) as src:

        # 获取源数据的地理变换信息、投影坐标系和像素大小
        src_transform = src.transform
        src_crs = src.crs
        src_res = src.res

        # 使用WKT字符串来定义目标数据的空间参考系
        dst_crs = rasterio.crs.CRS.from_wkt(
            'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]')

        # 计算目标数据的地理变换信息和像素大小，分辨率设置为源数据的分辨率
        dst_transform, width, height = calculate_default_transform(src_crs, dst_crs, src.width, src.height,dst_width=src.width, dst_height=src.height,*src.bounds)

        # 创建目标数据文件
        dst_profile = src.profile
        dst_profile.update({
            'crs': dst_crs,
            'transform': dst_transform,
            'width': width,
            'height': height,
            'dtype': 'int16',  # 设置数据类型为int16
            'nodata': -9999,  # 设置NoData值为-9999
        })

        with rasterio.open(dst, 'w', **dst_profile) as dst:

            # 进行投影变换
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )

if __name__ == "__main__":
    # datas_path = r"G:\projects\datas\ChengGuiData\dem"
    #
    # for dem_data in os.listdir(datas_path):
    #     src = os.path.join(datas_path, dem_data)
    #     dst = os.path.join(datas_path, dem_data.split(".tif")[0]+"_new.tif")
    #
    #     transform_data(src, dst)

    transform_data(r"G:\projects\datas\ChengGuiData\dem\Hancheng_s.tif", r"G:\projects\datas\ChengGuiData\dem\Hancheng_s_new.tif")