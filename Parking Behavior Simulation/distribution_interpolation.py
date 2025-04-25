import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d

# 读取数据
file_path = "/Users/fujunhan/Desktop/RA/my_research_related_materials/plan1.11/parking_data.xlsx"
df = pd.read_excel(file_path)

# 初始化 Kaplan-Meier 模型
kmf_idle = KaplanMeierFitter()
kmf_occupied = KaplanMeierFitter()

# 设置时间截断阈值
time_threshold = 75000

# 图 1: State = 0 (空闲状态)
state_0 = df[(df['State'] == 0) & (df['Lasting Time'] <= time_threshold)]
if not state_0.empty:
    durations_state_0 = state_0['Lasting Time']
    events_state_0 = [1] * len(state_0)
    kmf_idle.fit(durations=durations_state_0, event_observed=events_state_0)

# 图 2: State = 1 (占用状态)
state_1 = df[(df['State'] == 1) & (df['Lasting Time'] <= time_threshold)]
if not state_1.empty:
    durations_state_1 = state_1['Lasting Time']
    events_state_1 = [1] * len(state_1)
    kmf_occupied.fit(durations=durations_state_1, event_observed=events_state_1)

# 获取拟合的生存曲线
survival_idle = kmf_idle.survival_function_
survival_occupied = kmf_occupied.survival_function_

# 插值
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

# 创建时间点以进行插值
time_points = np.linspace(0, time_threshold, 1000)  # 生成 1000 个时间点

# 计算插值后的生存概率
survival_idle_values_interp = survival_idle_interp(time_points)
survival_occupied_values_interp = survival_occupied_interp(time_points)

# 创建子图
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# 绘制空闲状态的生存曲线
axs[0].plot(survival_idle.index, survival_idle.values, label='Idle State (Original)', marker='o')
axs[0].plot(time_points, survival_idle_values_interp, label='Idle State (Interpolated)', linestyle='--', color='green')
axs[0].set_xlabel('Time (seconds)')
axs[0].set_ylabel('Survival Probability')
axs[0].set_title('Idle State Survival Curve')
axs[0].legend()
axs[0].grid()

# 绘制占用状态的生存曲线
axs[1].plot(survival_occupied.index, survival_occupied.values, label='Occupied State (Original)', marker='o')
axs[1].plot(time_points, survival_occupied_values_interp, label='Occupied State (Interpolated)', linestyle='-', color = 'red')
axs[1].set_xlabel('Time (seconds)')
axs[1].set_ylabel('Survival Probability')
axs[1].set_title('Occupied State Survival Curve')
axs[1].legend()
axs[1].grid()

# 调整布局并显示图形
plt.tight_layout()
plt.show()