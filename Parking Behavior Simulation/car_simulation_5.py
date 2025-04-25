import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation

# 读取数据
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

# 图 2: State = 1 (占用状态)
state_1 = df[(df['State'] == 1) & (df['Lasting Time'] <= time_threshold)]  # 筛选占用状态并截断时间
if not state_1.empty:
    durations_state_1 = state_1['Lasting Time']
    events_state_1 = [1] * len(state_1)
    kmf_occupied.fit(durations=durations_state_1, event_observed=events_state_1)

# 获取拟合的生存曲线
survival_idle = kmf_idle.survival_function_
survival_occupied = kmf_occupied.survival_function_

# 初始化参数
max_minutes = 1440  # 模拟一天（1440 分钟）
num_parking_spots = 5  # 停车位数量
state_sequences = [[] for _ in range(num_parking_spots)]  # 保存每个停车位的状态序列
time_in_states = [0] * num_parking_spots  # 当前每个停车位的状态持续时间（分钟）
current_states = [0] * num_parking_spots  # 当前每个停车位的状态

rows = 1
cols = 5
fig, axs = plt.subplots(rows, cols, figsize=(16, 4))
axs = axs.flatten()  # 将二维数组展平，方便索引

lines = [axs[i].plot([], [], drawstyle="steps-post")[0] for i in range(num_parking_spots)]
for i in range(num_parking_spots):
    axs[i].set_xlim(0, max_minutes)
    axs[i].set_ylim(-0.1, 1.1)
    axs[i].set_xlabel("Time (minutes)")
    axs[i].set_ylabel("State (0: Idle, 1: Occupied)")
    axs[i].set_title(f"Parking Spot {i + 1} State Over Time")
    axs[i].grid()

survival_idle_interp = interp1d(
    survival_idle.index, 
    survival_idle.values.flatten(), 
    fill_value="extrapolate"
)
survival_occupied_interp = interp1d(
    survival_occupied.index, 
    survival_occupied.values.flatten(),
    fill_value="extrapolate"
)

# 更新函数使用插值函数
def update(frame):
    for i in range(num_parking_spots):
        # 将 time_in_state 转换为秒级时间
        time_in_seconds = time_in_states[i] * 60

        # 获取当前状态的生存概率（保持概率）
        if current_states[i] == 0:  # 空闲状态
            stay_prob = (survival_idle_interp(time_in_seconds) /
                          survival_idle_interp(time_in_seconds - 60)) if time_in_seconds >= 60 else 1
        else:  # 占用状态
            stay_prob = (survival_occupied_interp(time_in_seconds) /
                          survival_occupied_interp(time_in_seconds - 60)) if time_in_seconds >= 60 else 1

        # 决定是否状态转移
        a = np.random.rand()

        if a < stay_prob:
            # 状态保持，持续时间加 1
            time_in_states[i] += 1
        else:
            # 状态转移，切换状态并重置持续时间
            current_states[i] = 1 - current_states[i]  # 从空闲切换到占用，或反之
            time_in_states[i] = 1  # 初始化为 1 分钟，刚进入新状态
        
        # 更新状态序列并画图
        state_sequences[i].append(current_states[i])
        lines[i].set_data(range(len(state_sequences[i])), state_sequences[i])

    return lines

# 创建动画
ani = FuncAnimation(fig, update, frames=range(max_minutes), interval=100, repeat=False)

# 显示动画
plt.tight_layout()
plt.show()
