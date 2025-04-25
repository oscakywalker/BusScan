import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

# 假设你的数据存储在一个 Pandas DataFrame 中
file_path = "/Users/fujunhan/Desktop/RA/my_research_related_materials/plan1.11/parking_data.xlsx"
df = pd.read_excel(file_path)

# 筛选出 State 列为 1 的数据
state_1_data = df[df['State'] == 1]['Lasting Time']

# 使用 gaussian_kde 进行核密度估计
kde = gaussian_kde(state_1_data, bw_method=0.5)  # bw_method 调整平滑程度
x_vals = np.linspace(state_1_data.min(), state_1_data.max(), 1000)  # 在范围内生成 1000 个点
pdf_vals = kde(x_vals)  # 计算每个点的密度

# 绘制 PDF 曲线
plt.figure(figsize=(10, 6))
plt.plot(x_vals, pdf_vals, color='blue', lw=2, label="PDF (KDE)")
plt.fill_between(x_vals, pdf_vals, color='blue', alpha=0.3)  # 填充曲线下方区域
plt.title("PDF of Lasting Time for State = 1 (Gaussian KDE)", fontsize=16)
plt.xlabel("Lasting Time (seconds)", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.show()

