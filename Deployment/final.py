import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Polygon, Point
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import cv2
from datetime import timedelta
import tkinter as tk
from tkinter import messagebox
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

class ParkingRecord:
    def __init__(self, parking_id, route_id, position, geometry, kmf_idle, kmf_occupied, kde_idle, kde_occupied, time_threshold=75000):
        self.parking_id = parking_id
        self.route_id = route_id
        self.position = position
        self.geometry = geometry
        self.bus_pass_times = []
        self.kmf_idle = kmf_idle
        self.kmf_occupied = kmf_occupied
        self.kde_idle = kde_idle
        self.kde_occupied = kde_occupied
        self.time_threshold = time_threshold
        self.state = 0  # 初始状态为空闲
        self.remaining_time = 0  # 记录当前状态剩余时间
        self.bus_pass_times = []  # 存储 datetime 对象
        self.state_change_times = [] 
    
    def generate_random_from_kde(self, kde, size=1):
        """
        从 KDE 分布生成随机时间。
        """
        x_vals = np.linspace(0, self.time_threshold, 1000)
        pdf_vals = kde(x_vals)
        cdf_vals = np.cumsum(pdf_vals)
        cdf_vals /= cdf_vals[-1]
        inverse_cdf = interp1d(cdf_vals, x_vals, bounds_error=False, fill_value=(x_vals[0], x_vals[-1]))
        uniform_random = np.random.rand(size)
        return inverse_cdf(uniform_random)[0]
    
    def update_state(self, current_time):
        """
        更新停车位的状态（空闲或占用）。
        """
        if self.remaining_time > 0:
            self.remaining_time -= 1
        else:
            self.state = 1 - self.state  # 切换状态
            self.state_change_times.append(current_time)
            random_time = self.generate_random_from_kde(self.kde_idle if self.state == 0 else self.kde_occupied)
            self.remaining_time = max(1, int(random_time / 60))  # 确保至少持续 1 分钟
            print(f"Parking ID: {self.parking_id}, State: {self.state}, Remaining Time: {self.remaining_time}")
        return self.state
    
    def get_color(self):
        """
        根据当前状态返回颜色。
        """
        return 'purple' if self.state == 1 else 'green'  # 被占用为紫色，空闲为绿色
    
    def add_bus_pass_time(self, current_time):
        """
        记录巴士经过的时间。
        """
        self.bus_pass_times.append(current_time)

# ---------------------------- UI 配置 ----------------------------

# 全局变量存储区域信息
zone_mapping = {}  # 区域名称 -> 区域类型
zone_coordinates = {}  # 区域名称 -> (x_min, x_max, y_min, y_max)

# 定义添加区域的函数
def add_zone():
    try:
        zone_type = zone_type_entry.get()
        zone_name = zone_name_entry.get()
        x_min = float(x_min_entry.get())
        x_max = float(x_max_entry.get())
        y_min = float(y_min_entry.get())
        y_max = float(y_max_entry.get())

        # 检查区域名称是否已存在
        if zone_name in zone_mapping:
            messagebox.showerror("Error", f"Zone '{zone_name}' already exists!")
            return

        # 添加区域信息
        zone_mapping[zone_name] = zone_type
        zone_coordinates[zone_name] = (x_min, x_max, y_min, y_max)

        # 清空输入框
        zone_type_entry.delete(0, tk.END)
        zone_name_entry.delete(0, tk.END)
        x_min_entry.delete(0, tk.END)
        x_max_entry.delete(0, tk.END)
        y_min_entry.delete(0, tk.END)
        y_max_entry.delete(0, tk.END)

        messagebox.showinfo("Success", f"Zone '{zone_name}' added successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# 定义获取区域的函数
def get_zone(point):
    x, y = point.x, point.y
    for zone_name, (x_min, x_max, y_min, y_max) in zone_coordinates.items():
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return zone_name
    return "Unknown"

# 创建 UI 窗口
def create_ui():
    global zone_type_entry, zone_name_entry, x_min_entry, x_max_entry, y_min_entry, y_max_entry

    root = tk.Tk()
    root.title("Zone Configuration")

    # 创建输入框和标签
    tk.Label(root, text="Zone Type (CBD/Residential/Commute):").grid(row=0, column=0)
    zone_type_entry = tk.Entry(root)
    zone_type_entry.grid(row=0, column=1)

    tk.Label(root, text="Zone Name (e.g., Downtown):").grid(row=1, column=0)
    zone_name_entry = tk.Entry(root)
    zone_name_entry.grid(row=1, column=1)

    tk.Label(root, text="X Min:").grid(row=2, column=0)
    x_min_entry = tk.Entry(root)
    x_min_entry.grid(row=2, column=1)

    tk.Label(root, text="X Max:").grid(row=3, column=0)
    x_max_entry = tk.Entry(root)
    x_max_entry.grid(row=3, column=1)

    tk.Label(root, text="Y Min:").grid(row=4, column=0)
    y_min_entry = tk.Entry(root)
    y_min_entry.grid(row=4, column=1)

    tk.Label(root, text="Y Max:").grid(row=5, column=0)
    y_max_entry = tk.Entry(root)
    y_max_entry.grid(row=5, column=1)

    # 创建添加区域按钮
    tk.Button(root, text="Add Zone", command=add_zone).grid(row=6, column=0, columnspan=2)

    # 运行 UI
    root.mainloop()

# ---------------------------- 基础配置 ----------------------------
image_path = r'/Users/fujunhan/Desktop/Melbourne.jpg'
parking_interval = 0.5  # 停车位间隔(单位：地图单位)
detection_radius = 0.05  # 检测半径(单位：地图单位)
output_excel_path = r'/Users/fujunhan/Desktop/parking_bus_pass_times.xlsx'
file_path = "/Users/fujunhan/Desktop/RA/my_research_related_materials/plan1.11/parking_data.xlsx"
df = pd.read_excel(file_path)

# 初始化 Kaplan-Meier 模型
kmf_idle = KaplanMeierFitter()
kmf_occupied = KaplanMeierFitter()
# 设置时间截断阈值
time_threshold = 75000

# 图 1: State = 0 (空闲状态)
state_0 = df[(df['State'] == 0) & (df['Lasting Time'] <= time_threshold)]  # 筛选空闲状态并截断时间
if not state_0.empty:
    durations_state_0 = state_0['Lasting Time']
    events_state_0 = [1] * len(state_0)
    kmf_idle.fit(durations=durations_state_0, event_observed=events_state_0)
    kde_idle = gaussian_kde(durations_state_0, bw_method=0.5)  # KDE 计算空闲状态的 PDF

# 图 2: State = 1 (占用状态)
state_1 = df[(df['State'] == 1) & (df['Lasting Time'] <= time_threshold)]  # 筛选占用状态并截断时间
if not state_1.empty:
    durations_state_1 = state_1['Lasting Time']
    events_state_1 = [1] * len(state_1)
    kmf_occupied.fit(durations=durations_state_1, event_observed=events_state_1)
    kde_occupied = gaussian_kde(durations_state_1, bw_method=0.5)  # KDE 计算占用状态的 PDF

# 获取拟合的生存曲线
survival_idle = kmf_idle.survival_function_
survival_occupied = kmf_occupied.survival_function_
# ---------------------------- 初始化系统 ----------------------------
# 1. 加载并处理地图
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_image, 100, 200)
min_line_length = 100  # 最小直线长度
max_line_gap = 20      # 同一条直线上允许的最大间隙
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=150, 
                        minLineLength=min_line_length, maxLineGap=max_line_gap)

# 2. 创建城市边界
boundary = Polygon([(0,0), (60,0), (60,60), (0,60)])
gdf_city = gpd.GeoDataFrame(geometry=[boundary])

# 3. 生成巴士线路
lines_list = []
if lines is not None:
    h, w = image.shape[:2]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_geom = LineString([(x1/w*60, y1/h*60), (x2/w*60, y2/h*60)])
        lines_list.append(line_geom)
gdf_lines = gpd.GeoDataFrame(geometry=lines_list)

# 4. 生成停车位组
create_ui()
print("Zone Mapping:", zone_mapping)
print("Zone Coordinates:", zone_coordinates)

parking_records = []
for rid, route in enumerate(gdf_lines.geometry):
    length = route.length
    start_point = route.interpolate(0)  # 获取线段起点
    start_zone = get_zone(start_point)  # 判断起点所在区域
    zone_type = zone_mapping[start_zone]
    if zone_type == "CBD":
        num_parking_spaces = 6
    elif zone_type == "Residential":
        num_parking_spaces = 4
    elif zone_type == "Commute":
        num_parking_spaces = 0
    else:
        num_parking_spaces = 0
    for i in range(num_parking_spaces):
        dist = i * length / (num_parking_spaces - 1)
        point = route.interpolate(dist)
        parking_record = ParkingRecord(
            parking_id=f"Route_{rid}_Pos_{dist:.1f}",
            route_id=rid,
            position=dist,
            geometry=point,
            kmf_idle=kmf_idle,
            kmf_occupied=kmf_occupied,
            kde_idle=kde_idle,
            kde_occupied=kde_occupied
        )
        parking_records.append(parking_record)
gdf_parking = gpd.GeoDataFrame([record.__dict__ for record in parking_records])

# 5. 时间配置
time_intervals = pd.date_range("2023-01-01", periods=1440, freq='1T')  # 全天每分钟

# ---------------------------- 巴士系统 ----------------------------
# 区域配置
zone_config = {
    'CBD': {'peak': (5,10), 'off': (10,20)},
    'Residential': {'peak': (15,25), 'off': (20,30)},
    'Commute': {'peak': (5,10), 'off': (10,20)}
}

# 初始化巴士
bus_distribution = []
num_buses_per_route = 3  # 每条线路的巴士数量

for rid, route in enumerate(gdf_lines.geometry):
    route_length = route.length  # 当前线路的长度
    for i in range(num_buses_per_route):  # 每辆巴士的位置
        # 均匀分布计算位置：1/4, 1/2, 3/4
        start_position = (i + 1) * route_length / (num_buses_per_route + 1)  
        start_point = route.interpolate(start_position)  # 通过插值计算点位置

        # 获取区域类型和巴士的初始速度
        zone_type = zone_mapping[get_zone(start_point)]
        speed = random.uniform(*zone_config[zone_type]['off'])  # 获取区域的离峰速度范围
        
        # 添加巴士信息
        bus_distribution.append({
            'route': rid,
            'speed': speed,
            'position': start_position,
            'direction': 1,  # 初始方向为正向
            'geometry': start_point  # 起始位置的几何坐标
        })

# ---------------------------- 动画系统 ----------------------------
fig, ax = plt.subplots(figsize=(12, 8))

def update(frame):
    ax.clear()
    current_time = time_intervals[frame]
    
    # 更新巴士位置
    for bus in bus_distribution:
        bus_coord = bus['geometry']
        zone =  zone_mapping[get_zone(bus_coord)]
        zone_type = zone_config[zone]
        speed_range = zone_type['peak'] if current_time.hour in range(6, 10) or current_time.hour in range(16, 20) else zone_type['off']
        bus['speed'] = random.uniform(*speed_range)  # 更新速度
        delta = bus['speed'] * 1/60  # 每分钟移动距离
        bus['position'] += delta * bus['direction']
        
        route = gdf_lines.geometry[bus['route']]
        # 处理边界反弹
        if bus['position'] > route.length:
            bus['position'] = 2*route.length - bus['position']
            bus['direction'] *= -1
        elif bus['position'] < 0:
            bus['position'] = -bus['position']
            bus['direction'] *= -1
            
        bus['geometry'] = route.interpolate(bus['position'])
    
    # 检测巴士经过
    bus_points = [bus['geometry'] for bus in bus_distribution]
    gdf_buses = gpd.GeoDataFrame(geometry=bus_points)
    
    # 检测哪些停车位组有巴士经过
    buffers = gdf_parking.geometry.buffer(detection_radius)
    gdf_parking_buffers = gpd.GeoDataFrame(gdf_parking, geometry=buffers) 

    occupied = gpd.sjoin(gdf_parking_buffers, gdf_buses, predicate='intersects')
    
    # 记录巴士经过的时间
    for idx in occupied.index.unique():
        parking_record = parking_records[idx]
        parking_record.add_bus_pass_time(current_time)
    
    for record in parking_records:
        record.update_state(current_time)
    # 可视化
    gdf_city.plot(ax=ax, color='lightgray', edgecolor='k', alpha=0.5)
    gdf_lines.plot(ax=ax, color='blue', linewidth=1)
    
    # 绘制停车位组
    for record in parking_records:
        color = record.get_color()  # 获取当前状态的颜色
        ax.scatter(record.geometry.x, record.geometry.y, color=color, s=50, marker='s')
    
    # 绘制巴士
    gdf_buses.plot(ax=ax, color='red', markersize=50, marker='o', label='Buses')
    
    # 绘制被经过的停车位组
    if not occupied.empty:
        gdf_parking.loc[occupied.index].plot(ax=ax, color='orange', markersize=50, marker='s', label='Bus Passing')
    
    ax.scatter([], [], color='purple', s=50, marker='s', label='Occupied')
    ax.scatter([], [], color='green', s=50, marker='s', label='Empty')
    
    ax.set_title(f"Time: {current_time.strftime('%H:%M')}")
    plt.xlim(0, 60)
    plt.ylim(0, 60)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# ---------------------------- 运行系统 ----------------------------
# 先运行 UI 界面
create_ui()

# 再运行动画系统
ani = FuncAnimation(fig, update, frames=len(time_intervals), interval=50, repeat=False)

# ---------------------------- 生成报告 ----------------------------
plt.tight_layout()
plt.show()
report_data = []
all_time_diffs = []  # 存储所有有效时间差

for record in parking_records:
    # 将巴士经过时间排序以便查找
    bus_pass_times_sorted = sorted(record.bus_pass_times)
    time_diffs = []
    
    # 遍历每个状态变化时间
    for change_time in record.state_change_times:
        # 找到变化时间后的第一个巴士经过时间
        later_bus_times = [t for t in bus_pass_times_sorted if t > change_time]
        if later_bus_times:
            next_bus_time = later_bus_times[0]
            delta = next_bus_time - change_time
            time_diffs.append(delta.total_seconds() / 60)  # 转换为分钟
            
    # 汇总数据
    if time_diffs:
        avg_time_diff = sum(time_diffs) / len(time_diffs)
    else:
        avg_time_diff = None
    
    # 记录到报告
    report_data.append({
        'Parking ID': record.parking_id,
        'Avg Time Diff (min)': avg_time_diff,
        'State Changes': len(record.state_change_times),
        'Bus Pass Count': len(record.bus_pass_times)
    })
    
    # 收集全局时间差
    all_time_diffs.extend(time_diffs)

# 计算全局平均值
global_avg = sum(all_time_diffs) / len(all_time_diffs) if all_time_diffs else None
print(f"全局平均时间差：{global_avg:.2f} 分钟")

# 导出到 Excel
pd.DataFrame(report_data).to_excel(output_excel_path, index=False)

print(f"报告已生成至：{output_excel_path}")