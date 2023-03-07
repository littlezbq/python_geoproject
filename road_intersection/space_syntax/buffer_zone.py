import geopandas as gpd
from shapely.geometry import Point
import pyproj

# 读取道路路网和道路交叉口位置分布
roads = gpd.read_file('./test_data/road.shp')
intersections = gpd.read_file('./test_data/intersection.shp')

# 选取需要分析的交叉口
selected_intersections = intersections.loc[intersections['selected'] == True]

# 设置缓冲区半径，单位为米
buffer_radius = 200

# 将坐标系转换为UTM，单位为米
utm = pyproj.Proj(roads.crs)
lonlat = pyproj.Proj(init='epsg:4326')
selected_intersections = selected_intersections.to_crs(utm)

# 对每个交叉口进行缓冲区分析，找到半径范围内的道路段
buffered_roads = []
for i, intersection in selected_intersections.iterrows():
    center = intersection['geometry']
    buffer = center.buffer(buffer_radius)
    buffered_roads.append(buffer)

selected_roads = roads.loc[roads.intersects(buffered_roads)]

# 使用空间查询找到所有与道路段相交的区域
selected_areas = gpd.sjoin(roads, selected_roads, how='inner', op='intersects')

selected_areas.to_file("./test_data/selected_areas.shp")
