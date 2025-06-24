import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Polygon
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import cv2

# 1. 加载地图
image_path = r'/Users/fujunhan/Desktop/RA_traffic simulation/Codes/Bus route map/Chicago.jpg'
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_image, 100, 100)

# 2. 检测道路线
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=150, 
                       minLineLength=100, maxLineGap=10)

# 3. 创建城市边界
boundary = Polygon([(0, 0), (60, 0), (60, 60), (0, 60)])
gdf_city = gpd.GeoDataFrame(geometry=[boundary])

# 4. 转换道路坐标
lines_list = []
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        h, w = image.shape[:2]
        new_coords = [
            (int((x1/w)*60), int(60-(y1/h)*60)),
            (int((x2/w)*60), int(60-(y2/h)*60))
        ]
        lines_list.append(LineString(new_coords))
gdf_lines = gpd.GeoDataFrame(geometry=lines_list)

# 5. 交通参数配置（基于研究论文）
# 速度区间（km/h）
zone_speed_intervals = {
    'CBD': {  # 低速道路
        'peak_morning': (15, 25),
        'peak_evening': (15, 25),
        'off_peak': (30, 40)
    },
    'Residential': {  # 低速道路
        'peak_morning': (15, 25),
        'peak_evening': (15, 25),
        'off_peak': (30, 40)
    },
    'Commute': {  # 高速道路
        'peak_morning': (40, 60),
        'peak_evening': (40, 60),
        'off_peak': (50, 70)
    }
}

# 拥堵因子（Gao et al. 2020研究结论）
congestion_factors = {
    'normal': 1.0,
    'rainy': {
        'high_speed': 0.94,  # 高速路-雨天降速6%
        'low_speed': 0.97    # 低速路-雨天降速3%
    },
    'snowy': {
        'high_speed': 0.15,  # 高速路-雪天降速15%
        'low_speed': 0.10    # 低速路-雪天降速10%
    }
}

# 道路类型映射
road_type_mapping = {
    'CBD': 'low_speed',
    'Residential': 'low_speed',
    'Commute': 'high_speed'
}

# 6. 天气系统配置
weather_schedule = {
    20: 'snowy',    # 第20帧：降雪
    30: 'normal',   # 第30帧：恢复正常
    35: 'snowy',    # 第35帧：二次降雨
    45: 'normal'
}

weather_color = {
    'normal': 'red',
    'rainy': 'dodgerblue',
    'snowy': 'grey'
}

# 7. 区域划分函数
def get_zone(coord):
    x, y = coord
    if x <= 20 and y <= 30: return 'Sunset'
    elif 20 <= x <= 40 and y <= 30: return 'Pacifics'
    elif 40 <= x <= 60 and y <= 30: return 'Downtown'
    elif x <= 30 and y >= 30: return 'twin_peak'
    elif 30 <= x and 30 <= y <= 40: return 'Russian_hill'
    else: return 'Excelsion'

zone_mapping = {
    'Downtown': 'CBD',
    'Russian_hill': 'CBD',
    'Excelsion': 'Residential',
    'twin_peak': 'Commute',
    'Pacifics': 'Commute',
    'Sunset': 'Residential'
}

# 8. 初始化巴士
time_intervals = pd.date_range("2023-01-01", periods=48, freq='30T')
bus_distribution = []
current_weather = 'normal'

# 动态分配巴士（高速路分配更多车辆）
sorted_routes = sorted(enumerate(gdf_lines.geometry), 
                      key=lambda x: x[1].length, reverse=True)  # 按长度降序
route_bus_counts = {}
for i, (idx, route) in enumerate(sorted_routes):
    route_type = road_type_mapping[zone_mapping[get_zone(route.coords[0])]]
    base_buses = 3 if route_type == 'high_speed' else 1  # 高速路初始3辆
    route_bus_counts[idx] = base_buses + int(i * 0.5)  # 长度加权

# 生成巴士初始位置
for route_idx, route in enumerate(gdf_lines.geometry):
    for _ in range(route_bus_counts[route_idx]):
        pos = route.interpolate(random.random())
        zone = get_zone((pos.x, pos.y))
        zone_type = zone_mapping[zone]
        speed = random.uniform(*zone_speed_intervals[zone_type]['off_peak'])
        
        bus_distribution.append({
            'route_index': route_idx,
            'speed': speed,
            'direction': 1,
            'geometry': pos
        })
# 定义获取时间段的函数
def get_time_period(current_time):
    hour = current_time.hour
    if 7 <= hour < 9:
        return 'peak_morning'
    elif 17 <= hour < 19:
        return 'peak_evening'
    else:
        return 'off_peak'
# 9. 巴士运动逻辑
def update_bus_positions(bus_data, current_time, weather_state):
    time_period = get_time_period(current_time)
    
    for bus in bus_data:
        # 获取道路类型和基础速度
        route = gdf_lines.geometry[bus['route_index']]
        zone = get_zone((bus['geometry'].x, bus['geometry'].y))
        road_type = road_type_mapping[zone_mapping[zone]]
        base_speed = random.uniform(*zone_speed_intervals[zone_mapping[zone]][time_period])
        
        # 应用天气影响
        if weather_state == 'normal':
            bus['speed'] = base_speed
        else:
            factor = congestion_factors[weather_state][road_type]
            bus['speed'] = base_speed * factor
        
        # 计算新位置
        distance = bus['speed'] * 0.5  # 30分钟行程
        current_pos = route.project(bus['geometry'])
        new_pos = current_pos + distance * bus['direction']
        
        # 处理路线边界
        if new_pos > route.length or new_pos < 0:
            bus['direction'] *= -1
            new_pos = max(0, min(new_pos, route.length))
        
        bus['geometry'] = route.interpolate(new_pos)

# 10. 动画渲染
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), extent=[0, 60, 0, 60])

def update(frame):
    global current_weather
    if frame in weather_schedule:
        current_weather = weather_schedule[frame]
    
    ax.clear()
    current_time = time_intervals[frame]
    update_bus_positions(bus_distribution, current_time, current_weather)
    
    # 绘制元素
    gdf_city.plot(ax=ax, color='none', edgecolor='black', alpha=0.5)
    for idx, route in enumerate(gdf_lines.geometry):
        linewidth = 3 if road_type_mapping[zone_mapping[get_zone(route.coords[0])]] == 'high_speed' else 1
        gpd.GeoSeries([route]).plot(ax=ax, color='blue', linewidth=linewidth)
    
    buses = gpd.GeoDataFrame(bus_distribution, geometry='geometry')
    buses.plot(ax=ax, color=weather_color[current_weather], markersize=50, marker='o')
    
    # 天气标注
    #weather_icon = '☀️' if current_weather == 'normal' else '🌧️' if current_weather == 'rainy' else '❄️'
    ax.text(0.02, 0.95, f"{current_weather.upper()} | {current_time.strftime('%H:%M')}", 
            transform=ax.transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.xlim(0, 60)
    plt.ylim(0, 60)
    plt.title("Bus Traffic Simulation (High/Low Speed Roads)")
    plt.grid()

ani = FuncAnimation(fig, update, frames=len(time_intervals), interval=800)
plt.show()

# 输出统计
print(f"总线路: {len(gdf_lines)} (高速路: {sum(1 for r in gdf_lines.geometry if road_type_mapping[zone_mapping[get_zone(r.coords[0])]] == 'high_speed')})")
print(f"总巴士: {len(bus_distribution)}")
print("天气事件时间点:")
for f, w in sorted(weather_schedule.items()):
    print(f"帧{f:2d}: {time_intervals[f].strftime('%H:%M')} -> {w}")