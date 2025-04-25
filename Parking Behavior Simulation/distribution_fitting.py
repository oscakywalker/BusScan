import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# 读取数据
file_path = "/Users/fujunhan/Desktop/RA/my_research_related_materials/plan1.11/parking_data.xlsx"
df = pd.read_excel(file_path)

# 初始化 Kaplan-Meier 模型
kmf = KaplanMeierFitter()

# 创建两张图：State = 0 和 State = 1
plt.figure(figsize=(14, 6))

# 设置时间截断阈值
time_threshold = 75000

# 图 1: State = 0 (空闲状态)
plt.subplot(1, 2, 1)
state_0 = df[(df['State'] == 0) & (df['Lasting Time'] <= time_threshold)]  # 筛选空闲状态并截断时间
if not state_0.empty:
    # 合并所有车位的持续时间数据
    durations_state_0 = state_0['Lasting Time']
    events_state_0 = [1] * len(state_0)  # 设置事件发生标志为 1
    # 拟合并绘制 Kaplan-Meier 生存曲线
    kmf.fit(durations=durations_state_0, event_observed=events_state_0)
    kmf.plot_survival_function(label='Overall (Idle)', linestyle='--', color='green')

plt.title('Kaplan-Meier Survival Estimates (Idle parking lot)')
plt.xlabel('Time (seconds)')
plt.ylabel('Survival Probability')
plt.legend(loc='best')
plt.grid()

# 图 2: State = 1 (占用状态)
plt.subplot(1, 2, 2)
state_1 = df[(df['State'] == 1) & (df['Lasting Time'] <= time_threshold)]  # 筛选占用状态并截断时间
if not state_1.empty:
    # 合并所有车位的持续时间数据
    durations_state_1 = state_1['Lasting Time']
    events_state_1 = [1] * len(state_1)  # 设置事件发生标志为 1
    # 拟合并绘制 Kaplan-Meier 生存曲线
    kmf.fit(durations=durations_state_1, event_observed=events_state_1)
    kmf.plot_survival_function(label='Overall (Occupied)', linestyle='-', color='red')

plt.title('Kaplan-Meier Survival Estimates (Occupied parking lot)')
plt.xlabel('Time (seconds)')
plt.ylabel('Survival Probability')
plt.legend(loc='best')
plt.grid()

# 显示图形
plt.tight_layout()
plt.show()
