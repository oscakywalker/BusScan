import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from matplotlib.animation import FuncAnimation

# 读取数据
file_path = "/Users/fujunhan/Desktop/RA/my_research_related_materials/plan2.23/2020Melbourne_clean.csv"
df = pd.read_csv(file_path)

# 初始化 Kaplan-Meier 模型
kmf_idle = KaplanMeierFitter()
kmf_occupied = KaplanMeierFitter()

# 设置时间截断阈值
time_threshold = 75000

# 图 1: State = 0 (空闲状态)
state_0 = df[(df['State'] == 0) & (df['lastingtime'] <= time_threshold)]  # 筛选空闲状态并截断时间
if not state_0.empty:
    durations_state_0 = state_0['lastingtime']
    events_state_0 = [1] * len(state_0)
    kmf_idle.fit(durations=durations_state_0, event_observed=events_state_0)
    kde_idle = gaussian_kde(durations_state_0, bw_method=0.5)  # KDE 计算空闲状态的 PDF

# 图 2: State = 1 (占用状态)
state_1 = df[(df['State'] == 1) & (df['lastingtime'] <= time_threshold)]  # 筛选占用状态并截断时间
if not state_1.empty:
    durations_state_1 = state_1['lastingtime']
    events_state_1 = [1] * len(state_1)
    kmf_occupied.fit(durations=durations_state_1, event_observed=events_state_1)
    kde_occupied = gaussian_kde(durations_state_1, bw_method=0.5)  # KDE 计算占用状态的 PDF

# 获取拟合的生存曲线
survival_idle = kmf_idle.survival_function_
survival_occupied = kmf_occupied.survival_function_

# KDE 随机数生成函数
def generate_random_from_kde(kde, size):
    """
    根据 KDE 生成随机数。
    """
    x_vals = np.linspace(0, time_threshold, 1000)  # 在持续时间范围内生成点
    pdf_vals = kde(x_vals)  # 计算 PDF
    cdf_vals = np.cumsum(pdf_vals)  # 计算 CDF
    cdf_vals /= cdf_vals[-1]  # 归一化到 [0, 1]
    
    # 创建 CDF 的逆函数
    inverse_cdf = interp1d(cdf_vals, x_vals, bounds_error=False, fill_value=(x_vals[0], x_vals[-1]))
    
    # 生成随机数
    uniform_random = np.random.rand(size)  # 在 [0, 1] 上生成均匀分布的随机数
    sampled_random = inverse_cdf(uniform_random)  # 使用逆采样映射到目标分布
    return sampled_random

# 初始化参数
max_minutes = 1440  # 模拟时间（分钟）
num_parking_spots = 20  # 停车位数量
state_sequences = [[] for _ in range(num_parking_spots)]  # 保存每个停车位的状态序列
time_in_states = [0] * num_parking_spots  # 当前每个停车位的状态持续时间（分钟）
current_states = [0] * num_parking_spots  # 当前每个停车位的状态

# 用于记录所有停车位的状态变化
all_state_records = []

rows = 5
cols = 4
fig, axs = plt.subplots(rows, cols, figsize=(16, 12))
axs = axs.flatten()  # 将二维数组展平，方便索引

lines = [axs[i].plot([], [], drawstyle="steps-post")[0] for i in range(num_parking_spots)]
for i in range(num_parking_spots):
    axs[i].set_xlim(0, max_minutes)
    axs[i].set_ylim(-0.1, 1.1)
    axs[i].set_xlabel("Time (hours)")
    axs[i].set_ylabel("State (0: Idle, 1: Occupied)")
    axs[i].set_title(f"Parking Spot {i + 1} State Over Time")
    axs[i].grid()

    xticks = np.arange(0, max_minutes + 1, 3000)
    axs[i].set_xticks(xticks) 
    axs[i].set_xticklabels(xticks // 60)

# 更新函数使用 KDE 随机数生成
def update(frame):
    for i in range(num_parking_spots):
        # 如果当前状态的持续时间还没结束，状态保持
        if time_in_states[i] > 0:
            time_in_states[i] -= 1  # 剩余持续时间减少 1 分钟
        else:
            # 当前状态持续时间结束，记录状态及其随机生成的持续时间
            if frame > 0:  # 避免记录初始状态
                all_state_records.append({
                    "Parking Spot": i + 1,  # 停车位编号
                    "State": current_states[i],
                    "Lasting Time (minutes)": time_in_states[i],
                    "lastingtime": time_in_states[i] * 60
                })
            
            # 状态转移
            current_states[i] = 1 - current_states[i]  # 切换状态
            
            # 根据新的状态生成随机持续时间
            if current_states[i] == 0:  # 空闲状态
                random_time_in_seconds = generate_random_from_kde(kde_idle, size=1)[0]
            else:  # 占用状态
                random_time_in_seconds = generate_random_from_kde(kde_occupied, size=1)[0]
            
            # 转换为分钟，赋值给 time_in_states
            time_in_states[i] = max(1, int(random_time_in_seconds / 60))  # 用 int 确保是整数分钟，并且至少为 1 分钟

            # 记录当前状态的随机持续时间
            all_state_records.append({
                "Parking Spot": i + 1,  # 停车位编号
                "State": current_states[i],
                "Lasting Time (minutes)": time_in_states[i],  # 记录随机生成的持续时间
                "lastingtime": time_in_states[i] * 60
            })

        # 更新状态序列并画图
        state_sequences[i].append(current_states[i])
        lines[i].set_data(range(len(state_sequences[i])), state_sequences[i])
    return lines
    return lines

# 创建动画
ani = FuncAnimation(fig, update, frames=range(max_minutes), interval=100, repeat=False)

# 显示动画
plt.tight_layout()
plt.show()

# 将所有停车位的状态记录保存到同一张 Excel 表
all_state_records_df = pd.DataFrame(all_state_records)
all_state_records_df = all_state_records_df[all_state_records_df["Lasting Time (minutes)"] > 0]
output_file_path = "/Users/fujunhan/Desktop/parking_state_records.xlsx"
all_state_records_df.to_excel(output_file_path, index=False)
print(f"所有停车位的状态记录已保存到 {output_file_path}")