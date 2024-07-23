import pandas as pd
import matplotlib.pyplot as plt

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16

# 读取数据
data = pd.read_csv('num_u_final.csv')  # 更改为你的 CSV 文件路径

# 确保 'dataset/datasize' 是整数类型
data['dataset/num_u'] = data['dataset/num_u'].astype(int)

# 将 Source Name 映射到对应的模型名称
model_mapping = {
    'runnables/train_coso.py': 'CFPnet',
    'runnables/train_enc_dec.py': 'CRN',
    'runnables/train_gnet.py': 'G-Net',
    'runnables/train_rmsn.py': 'RMSNs'
}

# 添加模型名称列
data['Model Name'] = data['Source Name'].map(model_mapping)

# 打印调试信息
#print("原始数据：")
#print(data)

# 对 RMSE 进行归一化
rmse_min = data['Average encoder_text_rmse_all'].min()
rmse_max = data['Average encoder_text_rmse_all'].max()

data['Normalized RMSE'] = (data['Average encoder_text_rmse_all'] - rmse_min) / (rmse_max - rmse_min)

# 选择一个缩放因子直接缩放方差
scaling_factor = 0.005
data['Scaled StdDev RMSE'] = data['StdDev encoder_text_rmse_all'] * scaling_factor

# 打印缩放后的方差信息
#print("\n缩放后的数据：")
#print(data[['Model Name', 'dataset/datasize', 'Scaled StdDev RMSE']])

# 定义不同的线型和颜色
line_styles = {
    'CFPnet': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2, 'marker': 'o', 'markersize': 4},
    'CRN': {'color': 'green', 'linestyle': '--', 'linewidth': 2, 'marker': 's', 'markersize': 4},
    'G-Net': {'color': 'red', 'linestyle': '-.', 'linewidth': 2, 'marker': '^', 'markersize': 4},
    'RMSNs': {'color': 'purple', 'linestyle': ':', 'linewidth': 2, 'marker': 'd', 'markersize': 4}
}

# 对不同的 Model Name 进行分组，并绘图
fig, ax = plt.subplots(figsize=(10, 6))

for name, group in data.groupby('Model Name'):
    group = group.sort_values('dataset/num_u')  # 根据 datasize 排序
    ax.errorbar(group['dataset/num_u'], group['Normalized RMSE'], 
                yerr=group['Scaled StdDev RMSE'], label=name, 
                marker=line_styles[name]['marker'], capsize=5, 
                color=line_styles[name]['color'], linestyle=line_styles[name]['linestyle'],
                linewidth=line_styles[name]['linewidth'], markersize=line_styles[name]['markersize'])

# 设置 x 轴刻度仅显示数据中存在的值
ax.set_xticks(data['dataset/num_u'].unique())
ax.set_xticklabels([f'{int(x)}' for x in data['dataset/num_u'].unique()])

ax.set_xlabel('Number of U')
ax.set_ylabel('Normalized RMSE')

# 调整图例框架的大小、位置和边框透明度
#legend = ax.legend(bbox_to_anchor=(0.8, 1), loc='upper left', fontsize='large', title_fontsize='x-large')
#legend.get_frame().set_alpha(0.5)  # 设置图例边框透明度

# 增加图例内部元素之间的间距
#plt.setp(legend.get_texts(), fontsize='medium')  # 调整图例字体大小
#plt.setp(legend.get_title(), fontsize='large')  # 调整图例标题字体大小

ax.grid(True)

plt.show()
