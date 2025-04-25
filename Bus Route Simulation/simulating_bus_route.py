import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Polygon
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import cv2

# 1. 加载图片
image_path = r'/Users/fujunhan/Desktop/Chicago.jpg'
image = cv2.imread(image_path)

# 2. 转为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. 应用Canny边缘检测
edges = cv2.Canny(gray_image, threshold1=100, threshold2=100)

# 4. 使用霍夫变换检测图像中的直线
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=150, minLineLength=100, maxLineGap=10)

# 5. 创建SF城市边界的多边形
boundary_coords = [
    (0, 0), (60, 0), (60, 60), (0, 60)
]
boundary = Polygon(boundary_coords)  # 创建城市边界的多边形
gdf_city = gpd.GeoDataFrame(geometry=[boundary])

# 6. 绘制检测到的直线并放置于城市边界
lines_list = []  # 存放经过缩放的直线

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # 假设原始图像尺寸是 image.shape[:2] 获取高度和宽度
        orig_height, orig_width = image.shape[:2]
        
        # 线条的缩放
        new_x1 = int((x1 / orig_width) * 60)
        new_y1 = int(60 - (y1 / orig_height) * 60)
        new_x2 = int((x2 / orig_width) * 60)
        new_y2 = int(60 - (y2 / orig_height) * 60)
        
        # 创建LineString并加入列表
        line_geom = LineString([(new_x1, new_y1), (new_x2, new_y2)])
        lines_list.append(line_geom)

# 将所有线条放入GeoDataFrame
gdf_lines = gpd.GeoDataFrame(geometry=lines_list)

# 7. 可视化结果
fig, ax = plt.subplots(figsize=(8, 8))

# 显示原地图
ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), extent=[0, 60, 0, 60])

# 绘制城市边界
gdf_city.plot(ax=ax, color='none', edgecolor='black', alpha=0.5)  # 绘制城市边界

# 绘制缩放的线条
gdf_lines.plot(ax=ax, color='red', linewidth=2, label='Detected Lines')  # 绘制缩放的线条

time_intervals = pd.date_range("2023-01-01", periods=48, freq='30T')

zone_speed_intervals = {
    'CBD': {
        'peak_morning': (5, 10),
        'peak_evening': (5, 10),
        'off_peak': (10, 20)
    },
    'Residential': {
        'peak_morning': (15, 25),
        'peak_evening': (15, 25),
        'off_peak': (20, 30)
    },
    'Commute': {
        'peak_morning': (5, 10),
        'peak_evening': (5, 10),
        'off_peak': (10, 20)
    }
}

# 将区域映射到每个区
zone_mapping = {
    'Downtown': 'CBD',
    'Russian_hill': 'CBD',
    'Excelsion': 'Residential',
    'twin_peak': 'Commute',
    'Pacifics': 'Commute',
    'Sunset': 'Residential'
}

# 定义获取时间段的函数
def get_time_period(current_time):
    hour = current_time.hour
    if 7 <= hour < 9:
        return 'peak_morning'
    elif 17 <= hour < 19:
        return 'peak_evening'
    else:
        return 'off_peak'

def get_zone(coord):
    if coord[0] <= 20 and coord[1] <= 30:
        return 'Sunset'
    elif 20 <= coord[0] <= 40 and coord[1] <= 30:
        return 'Pacifics'
    elif 40 <= coord[0] <= 60 and coord[1] <= 30:
        return 'Downtown'
    elif coord[0] <= 30 and coord[1] >= 30:
        return 'twin_peak'
    elif 30 <= coord[0] and 30 <= coord[1] <= 40:
        return 'Russian_hill'
    else:
        return 'Excelsion'

# 初始化巴士分布，动态分配巴士数量
bus_distribution = []

# 获取每条线路的长度并排序
sorted_routes = sorted(enumerate(gdf_lines.geometry), key=lambda x: x[1].length)
min_buses = 1  # 最少分配1辆巴士
max_buses = 5  # 最多分配5辆巴士

# 动态分配巴士数量
route_bus_counts = {}
for i, (route_index, route) in enumerate(sorted_routes):
    # 根据线路在排序中的位置动态分配巴士数量
    buses_per_route = min_buses + int((max_buses - min_buses) * (i / (len(sorted_routes) - 1)))
    route_bus_counts[route_index] = buses_per_route

# 初始化巴士分布
for route_index, route in enumerate(gdf_lines.geometry):
    buses_per_route = route_bus_counts[route_index]  # 获取当前线路的巴士数量
    for _ in range(buses_per_route):
        # 仅在第一个时间步生成随机位置
        random_position = np.random.rand() * route.length
        bus_position = route.interpolate(random_position)
        bus_coord = (bus_position.x, bus_position.y)
        zone = get_zone(bus_coord)
        zone_type = zone_mapping[zone]
        speed_range = zone_speed_intervals[zone_type]['off_peak']  # 假设初始为非高峰期
        speed_value = random.uniform(*speed_range)  # 从速度区间随机取速度

        bus_distribution.append({
            'route_index': route_index,
            'speed': speed_value,
            'direction': 1,  # 方向: 1表示正向，-1表示反向
            'geometry': bus_position
        })

# 更新巴士位置（每个时间步）
def update_bus_positions(bus_data, current_time, time_interval_minutes=30):
    time_period = get_time_period(current_time)  # 获取当前时间段

    for bus in bus_data:
        # 重新计算巴士所在的区域
        bus_coord = (bus['geometry'].x, bus['geometry'].y)
        zone = get_zone(bus_coord)
        zone_type = zone_mapping[zone]

        # 重新计算速度范围
        speed_range = zone_speed_intervals[zone_type][time_period]
        bus['speed'] = random.uniform(*speed_range)  # 重新随机一个速度

        # 计算移动距离
        distance = bus['speed'] * (time_interval_minutes / 60.0)
        route = gdf_lines.geometry[bus['route_index']]
        route_length = route.length

        # 获取当前巴士的位置
        current_position = route.project(bus['geometry'])  # 计算在路线上的投影位置
        new_position = current_position + distance * bus['direction']

        # 检查是否超出路线边界
        if new_position > route_length:
            new_position = new_position - route_length
            bus['direction'] *= -1  # 反向行驶
        elif new_position < 0:
            new_position = -new_position
            bus['direction'] *= -1  # 反向行驶

        # 更新位置
        bus['geometry'] = route.interpolate(new_position)

# 动画更新
def update(frame):
    ax.clear()
    current_time = time_intervals[frame]

    # 更新巴士位置并调整速度
    update_bus_positions(bus_distribution, current_time)

    # 生成 GeoDataFrame
    bus_data = pd.DataFrame.from_records(bus_distribution, columns=['route_index', 'speed', 'direction', 'geometry'])
    bus_data['time'] = current_time  # 添加当前时间
    gdf_buses = gpd.GeoDataFrame(bus_data, geometry='geometry')  # 将 geometry 列作为几何列

    # 绘制地图
    gdf_city.plot(ax=ax, color='none', edgecolor='black', alpha=0.5, label='City Boundary')
    for route in gdf_lines.geometry:
        gpd.GeoSeries([route]).plot(ax=ax, color='blue', linewidth=2, label='Bus Routes')

    # 绘制巴士
    current_buses = gdf_buses[gdf_buses["time"] == current_time]
    current_buses["x"] = current_buses.geometry.x
    current_buses["y"] = current_buses.geometry.y
    current_buses.plot(ax=ax, color='red', marker='o', markersize=50, label='Buses')

    # 设置图像
    city_bounds = boundary.bounds
    plt.xlim(city_bounds[0], city_bounds[2])
    plt.ylim(city_bounds[1], city_bounds[3])
    plt.title(f'Bus Distribution at {current_time.strftime("%H:%M")}')
    plt.xlabel('X Coordinate (km)')
    plt.ylabel('Y Coordinate (km)')
    plt.grid()

    # 固定图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

# 创建动画
ani = FuncAnimation(fig, update, frames=len(time_intervals), interval=800, repeat=True)

plt.show()

total_lines = len(gdf_lines)
total_buses = sum(route_bus_counts.values())  # 累加每条线路的巴士数量

print(f"总线路数: {total_lines}")
print(f"初始总巴士数量: {total_buses}")