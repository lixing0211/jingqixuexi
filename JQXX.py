import os
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
# 屏蔽无关警告
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
# 保留原类名+别名，解决isinstance识别问题
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, LinearDiscriminantAnalysis as MDA
from sklearn.manifold import Isomap, TSNE
from sklearn.metrics import silhouette_score
import umap

# 自动创建输出文件夹，避免保存图片报错
os.makedirs("./data", exist_ok=True)
os.makedirs("./result", exist_ok=True)

print("加载MNIST 784维高维数据集...")
# 修正cache参数，data_home指定缓存目录
mnist = fetch_openml("mnist_784", version=1, cache=True, data_home="./data", parser="auto")
# 转换numpy数组，规避DataFrame行索引KeyError
X_all = mnist.data.to_numpy().astype(np.float32)
y_all = mnist.target.astype(int)

# 随机固定采样10000样本，保证复现
np.random.seed(42)
sample_idx = np.random.choice(len(X_all), size=10000, replace=False)
X = X_all[sample_idx]
y = y_all[sample_idx]

# 像素归一化+特征标准化
X = X / 255.0
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 绘图全局配置
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12

metrics_result = []

# ---------------------- 封装降维、绘图、评估函数 ----------------------
def run_dim_reduction(model, name, X_input, y_label):
    t_start = time.time()
    # 修复：现在LinearDiscriminantAnalysis已导入，可正常判断
    if isinstance(model, LinearDiscriminantAnalysis):
        embed = model.fit_transform(X_input, y_label)
    else:
        embed = model.fit_transform(X_input)
    t_cost = round(time.time() - t_start, 2)

    # 计算轮廓系数
    sil_score = silhouette_score(embed, y_label)

    # 绘制高清二维可视化图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embed[:, 0], embed[:, 1], c=y_label, s=3, cmap="tab10")
    plt.colorbar(scatter, label="数字标签 0-9")
    plt.title(f"{name} 降维二维可视化")
    plt.xlabel("维度1")
    plt.ylabel("维度2")
    plt.savefig(f"./result/{name}_vis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 仅原生PCA计算方差保留率，KPCA/MDA等无该属性
    var_ratio = None
    if isinstance(model, PCA):
        var_ratio = round(np.sum(model.explained_variance_ratio_[:2]) * 100, 1)

    print(f"{name} 训练耗时: {t_cost}s | 轮廓系数: {sil_score:.2f}")
    return {
        "算法": name,
        "轮廓系数": round(sil_score, 2),
        "训练耗时(s)": t_cost,
        "二维方差保留率(%)": var_ratio
    }

# ---------------------- 依次运行6种降维算法 ----------------------
# 1. PCA 线性无监督
pca = PCA(n_components=2, random_state=42)
metrics_result.append(run_dim_reduction(pca, "PCA", X_scaled, y))

# 2. MDA 线性有监督
mda = MDA(n_components=2)
metrics_result.append(run_dim_reduction(mda, "MDA", X_scaled, y))

# 3. KPCA RBF核，调整适配MNIST的gamma，提升聚类效果
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0005, random_state=42)
metrics_result.append(run_dim_reduction(kpca, "KPCA(RBF核)", X_scaled, y))

# 4. ISOMAP 全局流形
isomap = Isomap(n_components=2, n_neighbors=10)
metrics_result.append(run_dim_reduction(isomap, "ISOMAP", X_scaled, y))

# 5. t-SNE 局部概率流形
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
metrics_result.append(run_dim_reduction(tsne, "t-SNE", X_scaled, y))

# 6. UMAP 修复n_jobs警告，多核加速
umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
    n_jobs=-1  # 自动使用全部CPU核心，消除n_jobs警告
)
metrics_result.append(run_dim_reduction(umap_model, "UMAP", X_scaled, y))

# ---------------------- 保存量化指标表格，NaN替换为空方便Word查看 ----------------------
df_metric = pd.DataFrame(metrics_result)
df_metric = df_metric.fillna("")  # 把NaN填充为空字符串
df_metric.to_csv("./result/algorithm_metrics.csv", index=False, encoding="utf-8-sig")

print("\n全部实验完成！可视化图片与指标表格已保存至 ./result 文件夹")
print(df_metric)