import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def analyze_dataset():
    # 读取数据集
    df = pd.read_csv('./task2/dataset/comments_and_ratings_combined.csv')

    # 显示基本信息
    print("数据集基本信息:")
    print(df.info())
    print("\n前5条数据:")
    print(df.head())

    # 检查是否有缺失值
    print("\n缺失值统计:")
    print(df.isnull().sum())

    # 检查重复数据
    print("\n重复数据数量:", df.duplicated().sum())

    return df

def clean_text(text):
    """清洗单条文本"""
    if not isinstance(text, str):
        return ""
    
    # 去除URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 去除特殊字符和标点符号（保留中文标点）
    text = re.sub(r'[^\u4e00-\u9fa5\u3000-\u303f\uff00-\uff65a-zA-Z0-9，。！？、；：""''（）]', '', text)
    
    # 去除多余空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 去除空白字符前后的空格
    text = text.strip()
    
    return text

def clean_dataset(df):
    """清洗整个数据集"""
    print("开始数据清洗...")
    print(f"清洗前数据量: {len(df)}")
    
    # 删除完全重复的行
    df = df.drop_duplicates()
    print(f"去重后数据量: {len(df)}")
    
    # 清洗评论文本
    df['comment'] = df['comment'].apply(clean_text)
    
    # 删除清洗后为空的评论
    df = df[df['comment'].str.len() > 0]
    print(f"删除空评论后数据量: {len(df)}")
    
    # 删除评论长度过短的数据（比如少于5个字符）
    df = df[df['comment'].str.len() >= 2]
    print(f"删除短评论后数据量: {len(df)}")
    
    # 检查评分范围是否合理（1-10分）
    df = df[df['rating'].between(1, 10)]
    print(f"清洗后最终数据量: {len(df)}")
    
    # 显示清洗后的样例
    print("\n清洗后的数据样例:")
    print(df.head())
    
    return df

class AnimeDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, comments, ratings, tokenizer, max_length=512):
        self.comments = comments
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = str(self.comments[idx])
        rating = self.ratings[idx]

        # 使用tokenizer处理文本
        encoding = self.tokenizer(
            comment,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'rating': torch.tensor(rating, dtype=torch.float)
        }

def preprocess_data(df, test_size=0.2, random_state=42):
    """数据预处理函数"""
    print("开始数据预处理...")
    
    # 1. 标签归一化（将评分从1-10归一化到0-1）
    df['normalized_rating'] = (df['rating'] - 1) / 9
    
    # 2. 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 3. 划分训练集和测试集（分层抽样）
    # 首先划分训练集和测试集
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=pd.qcut(df['rating'], q=10, duplicates='drop')  # 使用评分分层
    )
    
    # 4. 创建数据集实例
    train_dataset = AnimeDataset(
        train_df['comment'].values,
        train_df['normalized_rating'].values,
        tokenizer
    )
    
    test_dataset = AnimeDataset(
        test_df['comment'].values,
        test_df['normalized_rating'].values,
        tokenizer
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 5. 检查数据集划分的评分分布
    print("\n训练集评分分布:")
    print(train_df['rating'].value_counts().sort_index())
    print("\n测试集评分分布:")
    print(test_df['rating'].value_counts().sort_index())
    
    return train_dataset, test_dataset, tokenizer

if __name__ == '__main__':
    # 读取原始数据
    df = analyze_dataset()
    
    # 清洗数据
    cleaned_df = clean_dataset(df)
    
    # 保存清洗后的数据
    cleaned_df.to_csv('./task2/dataset/cleaned_comments_and_ratings.csv', index=False)
    print("\n清洗后的数据已保存到cleaned_comments_and_ratings.csv")
    
    # 数据预处理
    train_dataset, test_dataset, tokenizer = preprocess_data(cleaned_df)
    
    # 保存tokenizer供后续使用
    tokenizer.save_pretrained('./task2/models/tokenizer')
    print("\nTokenizer已保存到./task2/models/tokenizer")
    
    # 创建DataLoader示例（实际训练时可能需要调整batch_size）
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 验证数据格式
    print("\n验证数据格式:")
    batch = next(iter(train_loader))
    print("Batch keys:", batch.keys())
    print("Input shape:", batch['input_ids'].shape)
    print("Attention mask shape:", batch['attention_mask'].shape)
    print("Rating shape:", batch['rating'].shape)
    
    # 测试BERT tokenizer的分词效果
    print("\nBERT tokenizer分词示例:")
    sample_text = "这是一个测试句子"
    tokens = tokenizer.tokenize(sample_text)
    print(f"原文: {sample_text}")
    print(f"分词结果: {tokens}")
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Token IDs: {token_ids}")
