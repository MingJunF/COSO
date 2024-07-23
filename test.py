import pandas as pd

# 读取CSV文件
df = pd.read_csv('CTMIMIC.csv')

# 预先检查一下CSV的列名
print(df.columns)

# 需要计算的列名列表
columns_to_aggregate = [
    'encoder_test_rmse_all',
    'decoder_test_rmse_2-step',
    'decoder_test_rmse_3-step',
    'decoder_test_rmse_4-step',
    'decoder_test_rmse_5-step',
    'decoder_test_rmse_6-step'
]

# 分组计算每个Source Name和dataset/num_covariates、dataset/num_confounder的平均值和标准差
agg_dict = {col: ['mean', 'std'] for col in columns_to_aggregate}
results = df.groupby(['Source Name']).agg(agg_dict)

# 展开多重索引列名
results.columns = ['_'.join(col) for col in results.columns]

# 重置索引，使得每个列名都可以正常显示
results.reset_index(inplace=True)

# 创建新的DataFrame以便存储最终结果
final_results = pd.DataFrame({
    'Source Name': results['Source Name'],
})

# 将每个列的均值和标准差添加到final_results中
for col in columns_to_aggregate:
    final_results[f'Average {col}'] = results[f'{col}_mean']
    final_results[f'StdDev {col}'] = results[f'{col}_std']

# 打印结果看看
print(final_results)

# 保存结果到新的CSV文件
final_results.to_csv('CTMIMICfinal.csv', index=False)

print("Completed and saved to Num_X_and_X_a_y_final.csv")
