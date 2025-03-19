import pandas as pd
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取三个CSV文件
df_high = pd.read_csv('task2/dataset/comments_and_ratings_high.csv')
df_medium = pd.read_csv('task2/dataset/comments_and_ratings_medium.csv')
df_low = pd.read_csv('task2/dataset/comments_and_ratings_low.csv')

# 确保每个数据框的rating列都是数值类型
for df in [df_high, df_medium, df_low]:
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# 合并所有数据框
df_combined = pd.concat([df_high, df_medium, df_low], ignore_index=True)

# 删除可能的空值
df_combined = df_combined.dropna()

# 保存合并后的CSV文件
df_combined.to_csv('task2/dataset/comments_and_ratings_combined.csv', index=False)

# 创建评分的频率分布图
plt.figure(figsize=(10, 6))
plt.hist(df_combined['rating'], bins=range(0, 11, 1), edgecolor='black', align='left')  # 使用plt.hist替代df.hist
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# 保存图片
plt.savefig('rating_distribution.png')
plt.close()

# 打印一些基本统计信息
print(f"总评论数: {len(df_combined)}")
print("\n评分统计信息:")
print(df_combined['rating'].describe())