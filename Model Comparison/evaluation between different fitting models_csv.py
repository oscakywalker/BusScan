import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines import WeibullFitter
from lifelines import LogLogisticFitter
from scipy.interpolate import interp1d

# 读取数据
file_path = "/Users/fujunhan/Desktop/RA/my_research_related_materials/plan2.23/2020Melbourne_clean.csv"
df = pd.read_csv(file_path)

# 设置时间截断阈值
time_threshold = 75000

def calculate_mse(model, durations, events, model_name=None):
    """
    计算 MSE（均方误差）用于评估模型对生存概率的预测偏差。

    参数:
        model: 训练好的生存分析模型（Kaplan-Meier, Weibull, Log-Logistic）。
        durations: 测试集中的持续时间或时间点。
        events: 测试集中的事件观察值（1 表示事件发生，0 表示未发生）。
        model_name (str): 指定模型，用于特殊情况处理（如 KMF）。

    返回:
        float: 平均均方误差（MSE）。
    """
    mse = 0
    N = len(durations)  # 测试数据大小

    for duration, event in zip(durations, events):
        if model_name == "KMF":  # 单独处理 Kaplan-Meier Fitter
            survival_prob = model(duration)  # 直接用 KMF predict 方法获取 S(t)
        else:  # Weibull 和 Log-Logistic
            survival_prob = model.survival_function_at_times(duration).values[0]

        if event == 1:
            # 事件发生时，真实值 Y = 1，理想生存概率应该接近于 0
            mse += (survival_prob - 0)**2
        else:
            # 事件未发生时，真实值 Y = 0，理想生存概率应该接近于 1
            mse += (survival_prob - 1)**2

    return mse / N

# 定义交叉验证分组
cv_splits = [
    (0, 200000),   # 第一组测试集：1-200
    (200000, 400000), # 第二组测试集：201-400
    (400000, 600000), # 第三组测试集：401-600
    (600000, 800000), # 第四组测试集：601-800
    (800000, len(df)) # 第五组测试集：801-最后
]

# 交叉验证函数
def cross_validation_mse(df, cv_splits, time_threshold):
    # 保存每个模型的对数似然值
    mse_scores = {
        "Kaplan-Meier (State=0)": [],
        "Kaplan-Meier (State=1)": [],
        "Weibull (State=0)": [],
        "Weibull (State=1)": [],
        "Log-Logistic (State=0)": [],
        "Log-Logistic (State=1)": []
    }
    
    for start, end in cv_splits:
        # 划分测试集和训练集
        test_data = df.iloc[start:end]
        train_data = pd.concat([df.iloc[:start], df.iloc[end:]])

        # 筛选 State = 0 数据
        state_0 = train_data[(train_data['State'] == 0) & (train_data['lastingtime'] <= time_threshold)]
        durations_state_0 = state_0['lastingtime']
        events_state_0 = [1] * len(state_0)

        # 筛选 State = 1 数据
        state_1 = train_data[(train_data['State'] == 1) & (train_data['lastingtime'] <= time_threshold)]
        durations_state_1 = state_1['lastingtime']
        events_state_1 = [1] * len(state_1)

        # 准备测试数据
        test_state_0 = test_data[(test_data['State'] == 0) & (test_data['lastingtime'] <= time_threshold)]
        test_durations_state_0 = test_state_0['lastingtime']
        test_events_state_0 = [1] * len(test_state_0)

        test_state_1 = test_data[(test_data['State'] == 1) & (test_data['lastingtime'] <= time_threshold)]
        test_durations_state_1 = test_state_1['lastingtime']
        test_events_state_1 = [1] * len(test_state_1)

        # 初始化和训练模型
        kmf_idle = KaplanMeierFitter()
        kmf_occupied = KaplanMeierFitter()
        wf_idle = WeibullFitter()
        llf_idle = LogLogisticFitter()
        wf_occupied = WeibullFitter()
        llf_occupied = LogLogisticFitter()

        # 训练模型
        kmf_idle.fit(durations_state_0, event_observed=events_state_0)
        kmf_occupied.fit(durations_state_1, event_observed=events_state_1)
        wf_idle.fit(durations_state_0, event_observed=events_state_0)
        llf_idle.fit(durations_state_0, event_observed=events_state_0)
        wf_occupied.fit(durations_state_1, event_observed=events_state_1)
        llf_occupied.fit(durations_state_1, event_observed=events_state_1)

        # Kaplan-Meier 插值
        survival_idle_interp = interp1d(
            kmf_idle.survival_function_.index,
            kmf_idle.survival_function_.values.flatten(),
            fill_value="extrapolate"
        )
        survival_occupied_interp = interp1d(
            kmf_occupied.survival_function_.index,
            kmf_occupied.survival_function_.values.flatten(),
            fill_value="extrapolate"
        )

        mse_scores["Kaplan-Meier (State=0)"].append(
            calculate_mse(survival_idle_interp, test_durations_state_0, test_events_state_0, "KMF")
        )
        mse_scores["Kaplan-Meier (State=1)"].append(
            calculate_mse(survival_occupied_interp, test_durations_state_1, test_events_state_1, "KMF")
        )
        mse_scores["Weibull (State=0)"].append(
            calculate_mse(wf_idle, test_durations_state_0, test_events_state_0)
        )
        mse_scores["Weibull (State=1)"].append(
            calculate_mse(wf_occupied, test_durations_state_1, test_events_state_1)
        )
        mse_scores["Log-Logistic (State=0)"].append(
            calculate_mse(llf_idle, test_durations_state_0, test_events_state_0)
        )
        mse_scores["Log-Logistic (State=1)"].append(
            calculate_mse(llf_occupied, test_durations_state_1, test_events_state_1)
        )

    # 计算每个模型的 MSE 平均值
    avg_mses = {model: np.mean(values) for model, values in mse_scores.items()}
    return avg_mses

# 调用交叉验证函数
avg_mses = cross_validation_mse(df, cv_splits, time_threshold)

# 打印结果
print("Average MSE Values (Cross-Validation):")
for model, avg_mse in avg_mses.items():
    print(f"{model}: {avg_mse}")
