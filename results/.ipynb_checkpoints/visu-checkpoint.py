import numpy as np
import matplotlib.pyplot as plt

# 模型名称
models = ["Random Forest", "Gradient Boosting", "Linear Regression"]

# 模型的评估指标 (示例数据)
rmse = [67000, 64500, 65000]  # 均方根误差
mae = [50000, 48000, 49000]   # 平均绝对误差
r2 = [0.85, 0.88, 0.82]       # R² 分数（为了可视化，调整比例）

# 位置
x = np.arange(len(models))
width = 0.25  # 每个柱状图的宽度

# 创建画布
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制柱状图
rects1 = ax.bar(x - width, rmse, width, label="RMSE", color="steelblue")
rects2 = ax.bar(x, mae, width, label="MAE", color="darkorange")
rects3 = ax.bar(x + width, np.array(r2) * 70000, width, label="R² (Scaled)", color="green")  # 调整 R² 以适应其他值

# 设置标题和轴标签
ax.set_xlabel("Models", fontsize=12)
ax.set_ylabel("Error Value", fontsize=12)
ax.set_title("Model Performance Comparison", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, rotation=15)

# 显示图例
ax.legend(title="Metric")

# 显示数值标签
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 轻微向上偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

# 显示网格线
ax.yaxis.grid(True, linestyle="--", alpha=0.7)

# 显示图表
plt.tight_layout()
plt.show()
