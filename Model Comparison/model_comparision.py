import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines import WeibullFitter
from lifelines import LogLogisticFitter

# 读取数据
file_path = "/Users/fujunhan/Desktop/RA/my_research_related_materials/plan1.11/parking_data.xlsx"
df = pd.read_excel(file_path)

# 划分训练集和测试集
train_data = df.iloc[:800]
test_data = df.iloc[800:]

# 设置时间截断阈值
time_threshold = 75000

# 筛选 State = 0 数据 (空闲状态)
state_0 = train_data[(train_data['State'] == 0) & (train_data['Lasting Time'] <= time_threshold)]
durations_state_0 = state_0['Lasting Time']
events_state_0 = [1] * len(state_0)  # 假设所有事件都发生

# 筛选 State = 1 数据 (占用状态)
state_1 = train_data[(train_data['State'] == 1) & (train_data['Lasting Time'] <= time_threshold)]
durations_state_1 = state_1['Lasting Time']
events_state_1 = [1] * len(state_1)  # 假设所有事件都发生

# 初始化 Kaplan-Meier 拟合器
kmf_idle = KaplanMeierFitter()
kmf_occupied = KaplanMeierFitter()

# Kaplan-Meier 分别拟合 State = 0 和 State = 1
kmf_idle.fit(durations_state_0, event_observed=events_state_0, label="Kaplan-Meier (State = 0)")
kmf_occupied.fit(durations_state_1, event_observed=events_state_1, label="Kaplan-Meier (State = 1)")

# 初始化 Weibull 和 Log-Logistic 拟合器
weibull_fitter_idle = WeibullFitter()
loglogistic_fitter_idle = LogLogisticFitter()
weibull_fitter_occupied = WeibullFitter()
loglogistic_fitter_occupied = LogLogisticFitter()

# 分别对 State = 0 进行 Weibull 和 Log-Logistic 拟合
weibull_fitter_idle.fit(durations_state_0, event_observed=events_state_0)
loglogistic_fitter_idle.fit(durations_state_0, event_observed=events_state_0)

# 分别对 State = 1 进行 Weibull 和 Log-Logistic 拟合
weibull_fitter_occupied.fit(durations_state_1, event_observed=events_state_1)
loglogistic_fitter_occupied.fit(durations_state_1, event_observed=events_state_1)

# --- 可视化 ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 创建 2x3 的子图网格
axes = axes.flatten()  # 将 2D 数组展平为 1D 方便索引

# 第一行：State = 0
# 子图 1: Kaplan-Meier (State = 0)
kmf_idle.plot_survival_function(ax=axes[0], ci_show=False, color='blue')
axes[0].set_title("Kaplan-Meier - State = 0", fontsize=14)
axes[0].set_xlabel("Time (seconds)", fontsize=12)
axes[0].set_ylabel("Survival Probability", fontsize=12)
axes[0].grid(alpha=0.5)

# 子图 2: Weibull (State = 0)
axes[1].plot(weibull_fitter_idle.survival_function_.index, weibull_fitter_idle.survival_function_.values,
             color='blue', label="Weibull (State = 0)")
axes[1].set_title("Weibull - State = 0", fontsize=14)
axes[1].set_xlabel("Time (seconds)", fontsize=12)
axes[1].set_ylabel("Survival Probability", fontsize=12)
axes[1].grid(alpha=0.5)

# 子图 3: Log-Logistic (State = 0)
axes[2].plot(loglogistic_fitter_idle.survival_function_.index, loglogistic_fitter_idle.survival_function_.values,
             color='blue', label="Log-Logistic (State = 0)")
axes[2].set_title("Log-Logistic - State = 0", fontsize=14)
axes[2].set_xlabel("Time (seconds)", fontsize=12)
axes[2].set_ylabel("Survival Probability", fontsize=12)
axes[2].grid(alpha=0.5)

# 第二行：State = 1
# 子图 4: Kaplan-Meier (State = 1)
kmf_occupied.plot_survival_function(ax=axes[3], ci_show=False, color='red')
axes[3].set_title("Kaplan-Meier - State = 1", fontsize=14)
axes[3].set_xlabel("Time (seconds)", fontsize=12)
axes[3].set_ylabel("Survival Probability", fontsize=12)
axes[3].grid(alpha=0.5)

# 子图 5: Weibull (State = 1)
axes[4].plot(weibull_fitter_occupied.survival_function_.index, weibull_fitter_occupied.survival_function_.values,
             color='red', label="Weibull (State = 1)")
axes[4].set_title("Weibull - State = 1", fontsize=14)
axes[4].set_xlabel("Time (seconds)", fontsize=12)
axes[4].set_ylabel("Survival Probability", fontsize=12)
axes[4].grid(alpha=0.5)

# 子图 6: Log-Logistic (State = 1)
axes[5].plot(loglogistic_fitter_occupied.survival_function_.index, loglogistic_fitter_occupied.survival_function_.values,
             color='red', label="Log-Logistic (State = 1)")
axes[5].set_title("Log-Logistic - State = 1", fontsize=14)
axes[5].set_xlabel("Time (seconds)", fontsize=12)
axes[5].set_ylabel("Survival Probability", fontsize=12)
axes[5].grid(alpha=0.5)

# 调整布局
plt.tight_layout()
plt.show()