import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# 读取数据
file_path = "/Users/fujunhan/Desktop/RA/my_research_related_materials/plan1.11/parking_data.xlsx"
df = pd.read_excel(file_path)

grouped = df.groupby("车位编号")

# 初始化 Kaplan-Meier 模型
kmf = KaplanMeierFitter()

# 创建两张图：State = 0 和 State = 1
plt.figure(figsize=(14, 6))

# 图 1: State = 0 (空闲状态)
plt.subplot(1, 2, 1)
state_0_empty = True  # 检查是否有数据
for parking_id, group in grouped:
    state_0 = group[group['State'] == 0]
    if not state_0.empty:
        state_0_empty = False
        #print(f"Parking {parking_id} - State 0 数据量: {len(state_0)}")
        # 如果State = 0 代表空闲停车位，设置event_observed为1（事件发生），表示空闲状态的持续时间
        kmf.fit(durations=state_0['Lasting Time'], event_observed=[1] * len(state_0))
        kmf.plot_survival_function(label=f'Parking {parking_id} (Idle)', linestyle='--')


plt.title('Kaplan-Meier Survival Estimates (Idle parking lot)')
plt.xlabel('Time (seconds)')
plt.ylabel('Survival Probability')
plt.legend(loc='best')
plt.grid()

# 图 2: State = 1 (占用状态)
plt.subplot(1, 2, 2)
state_1_empty = True  # 检查是否有数据
for parking_id, group in grouped:
    state_1 = group[group['State'] == 1]
    if not state_1.empty:
        state_1_empty = False
        #print(f"Parking {parking_id} - State 1 数据量: {len(state_1)}")
        # 如果State = 1 代表占用停车位，设置event_observed为1（事件发生），表示占用状态的持续时间
        kmf.fit(durations=state_1['Lasting Time'], event_observed=[1] * len(state_1))
        kmf.plot_survival_function(label=f'Parking {parking_id} (Occupied)', linestyle='-')


plt.title('Kaplan-Meier Survival Estimates (Occupied parking lot)')
plt.xlabel('Time (seconds)')
plt.ylabel('Survival Probability')
plt.legend(loc='best')
plt.grid()

# 显示图形
plt.tight_layout()
plt.show()
