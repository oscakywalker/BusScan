import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy.stats import gaussian_kde
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

# 1. 加载数据
file_path1 = "/Users/fujunhan/Desktop/parking_state_records.xlsx"  # 第一个文件路径
file_path2 = "/Users/fujunhan/Desktop/RA/my_research_related_materials/plan2.23/2020Melbourne_clean.csv"  # 第二个文件路径

df1 = pd.read_excel(file_path1)
df2 = pd.read_csv(file_path2)

# 2. 定义函数：计算生存函数和 KDE
def calculate_survival_and_kde(df, state, time_threshold):
    kmf = KaplanMeierFitter()
    state_data = df[(df['State'] == state) & (df['lastingtime'] <= time_threshold)]
    if not state_data.empty:
        durations = state_data['lastingtime']
        events = [1] * len(state_data)
        kmf.fit(durations=durations, event_observed=events)
        kde = gaussian_kde(durations, bw_method=0.5)
        return kmf.survival_function_, kde
    else:
        return None, None

# 3. 计算生存函数和 KDE
time_threshold = 75000

# 第一个文件
survival_idle1, kde_idle1 = calculate_survival_and_kde(df1, state=0, time_threshold=time_threshold)
survival_occupied1, kde_occupied1 = calculate_survival_and_kde(df1, state=1, time_threshold=time_threshold)

# 第二个文件
survival_idle2, kde_idle2 = calculate_survival_and_kde(df2, state=0, time_threshold=time_threshold)
survival_occupied2, kde_occupied2 = calculate_survival_and_kde(df2, state=1, time_threshold=time_threshold)

# 4. 绘制生存函数曲线
plt.figure(figsize=(12, 6))

# State=0
if survival_idle1 is not None and survival_idle2 is not None:
    plt.plot(survival_idle1.index, survival_idle1['KM_estimate'], label='State=0 (File 1)')
    plt.plot(survival_idle2.index, survival_idle2['KM_estimate'], label='State=0 (File 2)')

# State=1
if survival_occupied1 is not None and survival_occupied2 is not None:
    plt.plot(survival_occupied1.index, survival_occupied1['KM_estimate'], label='State=1 (File 1)')
    plt.plot(survival_occupied2.index, survival_occupied2['KM_estimate'], label='State=1 (File 2)')

plt.title('Survival Functions Comparison')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.legend()
plt.grid(True)
plt.show()

# 5. 相似度分析
def calculate_cosine_similarity(kde1, kde2):
    if kde1 is not None and kde2 is not None:
        x = np.linspace(0, time_threshold, 1000)
        pdf1 = kde1(x)
        pdf2 = kde2(x)
        return cosine_similarity([pdf1], [pdf2])[0][0]
    else:
        return None

def calculate_pearson_correlation(kde1, kde2):
    if kde1 is not None and kde2 is not None:
        x = np.linspace(0, time_threshold, 1000)  # 在时间范围内生成点
        pdf1 = kde1(x)  # 第一个数据的 PDF
        pdf2 = kde2(x)  # 第二个数据的 PDF
        # 计算皮尔逊相关系数
        correlation, _ = pearsonr(pdf1, pdf2)
        return correlation
    else:
        return None

# State=0 相似度
similarity_idle = calculate_cosine_similarity(kde_idle1, kde_idle2)
print(f"State=0 Cosine Similarity: {similarity_idle}")

# State=1 相似度
similarity_occupied = calculate_cosine_similarity(kde_occupied1, kde_occupied2)
print(f"State=1 Cosine Similarity: {similarity_occupied}")

pearson_idle = calculate_pearson_correlation(kde_idle1, kde_idle2)
print(f"State=0 Pearson Correlation: {pearson_idle}")

# State=1 相似度（皮尔逊相关系数）
pearson_occupied = calculate_pearson_correlation(kde_occupied1, kde_occupied2)
print(f"State=1 Pearson Correlation: {pearson_occupied}")