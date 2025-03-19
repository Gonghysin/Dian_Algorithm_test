import pandas as pd
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

# 读取CSV文件
df = pd.read_csv('task2/dataset/comments_and_ratings.csv')

# 确保rating列是数值类型
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna()  # 删除无效值

# 创建评分的频率分布图
plt.figure(figsize=(10, 6))
plt.hist(df['rating'], bins=range(0, 11, 1), edgecolor='black', align='left')
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# 保存图片
plt.savefig('single_rating_distribution.png')
plt.close()

# 打印统计信息
print(f"总评论数: {len(df)}")
print("\n评分统计信息:")
print(df['rating'].describe())