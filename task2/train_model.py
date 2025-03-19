import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig, AdamW
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os

# 确保模型保存目录存在
os.makedirs('task2/models', exist_ok=True)

class AnimeDataset(Dataset):
    """简化的数据集类"""
    def __init__(self, comments, ratings, tokenizer, max_length=128):
        self.comments = comments
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = str(self.comments[idx])
        rating = self.ratings[idx]

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

class SimpleRatingPredictor(nn.Module):
    """简化的评分预测模型"""
    def __init__(self, pretrained_model_path):
        super().__init__()
        config = BertConfig.from_json_file(f'{pretrained_model_path}/config.json')
        self.bert = BertModel.from_pretrained(
            pretrained_model_path,
            config=config,
            local_files_only=True
        )
        
        # 简单的回归头
        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, 1),
            nn.Sigmoid()  # 确保输出在0-1范围内
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.regressor(cls_output).squeeze(-1)

def load_data(batch_size=32, sample_ratio=1.0):
    """加载数据集"""
    print("加载数据...")
    
    # 加载训练集和验证集
    train_df = pd.read_csv('task2/dataset/train_set.csv')
    val_df = pd.read_csv('task2/dataset/val_set.csv')
    
    # 如果需要，进行采样
    if sample_ratio < 1.0:
        train_df = train_df.sample(frac=sample_ratio, random_state=42)
        val_df = val_df.sample(frac=sample_ratio, random_state=42)
        print(f"使用{sample_ratio*100:.1f}%的数据进行训练")
    
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained('task2/models/bert-base-chinese')
    
    # 创建数据集和数据加载器
    train_dataset = AnimeDataset(
        train_df['comment'].values,
        train_df['normalized_rating'].values,
        tokenizer
    )
    
    val_dataset = AnimeDataset(
        val_df['comment'].values,
        val_df['normalized_rating'].values,
        tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_and_evaluate(model, train_loader, val_loader, epochs=5, lr=2e-5, device='cuda'):
    """训练和评估模型"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    val_maes = []
    val_spearmans = []
    
    print("开始训练...")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ratings = batch['rating'].to(device)
            
            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 评估阶段
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                ratings = batch['rating'].to(device)
                
                predictions = model(input_ids, attention_mask)
                loss = criterion(predictions, ratings)
                val_loss += loss.item()
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(ratings.cpu().numpy())
        
        # 计算评估指标
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        mae = mean_absolute_error(all_labels, all_preds)
        val_maes.append(mae)
        
        spearman, _ = spearmanr(all_labels, all_preds)
        val_spearmans.append(spearman)
        
        # 打印评估结果
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, MAE: {mae:.4f}, Spearman: {spearman:.4f}")
    
    # 保存最终模型
    torch.save(model.state_dict(), 'task2/models/simple_model.pth')
    print("模型已保存到 task2/models/simple_model.pth")
    
    # 绘制训练过程
    plot_training_history(train_losses, val_losses, val_maes, val_spearmans)
    
    return model

def train_and_evaluate(model, train_loader, val_loader, epochs=5, lr=2e-5, device='cuda'):
    """训练和评估模型"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    val_maes = []
    val_spearmans = []
    val_mses = []  # 添加MSE记录
    val_rmses = []  # 添加RMSE记录
    
    # 记录最后一轮的预测结果
    final_preds = None
    final_labels = None
    
    print("开始训练...")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ratings = batch['rating'].to(device)
            
            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 评估阶段
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                ratings = batch['rating'].to(device)
                
                predictions = model(input_ids, attention_mask)
                loss = criterion(predictions, ratings)
                val_loss += loss.item()
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(ratings.cpu().numpy())
        
        # 保存最后一轮的预测结果
        if epoch == epochs - 1:
            final_preds = all_preds
            final_labels = all_labels
        
        # 计算评估指标
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 计算MSE和RMSE
        mse = mean_squared_error(all_labels, all_preds)
        rmse = np.sqrt(mse)
        val_mses.append(mse)
        val_rmses.append(rmse)
        
        mae = mean_absolute_error(all_labels, all_preds)
        val_maes.append(mae)
        
        spearman, _ = spearmanr(all_labels, all_preds)
        val_spearmans.append(spearman)
        
        # 打印评估结果
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, Spearman: {spearman:.4f}")
    
    # 保存最终模型
    torch.save(model.state_dict(), 'task2/models/simple_model.pth')
    print("模型已保存到 task2/models/simple_model.pth")
    
    # 绘制训练过程
    plot_training_history(train_losses, val_losses, val_maes, val_spearmans, val_mses, val_rmses, final_labels, final_preds)
    
    # 打印最终评估结果
    print("\n最终评估结果:")
    print(f"MSE: {val_mses[-1]:.4f}")
    print(f"RMSE: {val_rmses[-1]:.4f}")
    print(f"MAE: {val_maes[-1]:.4f}")
    print(f"Spearman相关系数: {val_spearmans[-1]:.4f}")
    
    return model

def plot_training_history(train_losses, val_losses, val_maes, val_spearmans, val_mses, val_rmses, all_labels=None, all_preds=None):
    """绘制更详细的训练历史图表"""
    plt.figure(figsize=(15, 10))
    
    # 1. MSE损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(val_mses, label='Val MSE')
    plt.title('Loss and MSE over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/MSE')
    plt.legend()
    
    # 2. MAE和RMSE曲线
    plt.subplot(2, 2, 2)
    plt.plot(val_maes, label='MAE')
    plt.plot(val_rmses, label='RMSE')
    plt.title('MAE and RMSE over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    
    # 3. 预测值vs真实值散点图
    if all_labels is not None and all_preds is not None:
        plt.subplot(2, 2, 3)
        plt.scatter(all_labels, all_preds, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.title('Predictions vs Ground Truth')
        plt.xlabel('True Ratings')
        plt.ylabel('Predicted Ratings')
    
    # 4. 评分分布对比
    if all_labels is not None and all_preds is not None:
        plt.subplot(2, 2, 4)
        plt.hist(all_labels, bins=20, alpha=0.5, label='True')
        plt.hist(all_preds, bins=20, alpha=0.5, label='Predicted')
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('task2/models/simple_training_history.png')
    plt.close()
    print("训练历史图表已保存到 task2/models/simple_training_history.png")

def train_and_evaluate(model, train_loader, val_loader, epochs=5, lr=2e-5, device='cuda'):
    """训练和评估模型"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    val_maes = []
    val_spearmans = []
    
    # 记录最后一轮的预测结果
    final_preds = None
    final_labels = None
    
    print("开始训练...")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ratings = batch['rating'].to(device)
            
            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 评估阶段
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                ratings = batch['rating'].to(device)
                
                predictions = model(input_ids, attention_mask)
                loss = criterion(predictions, ratings)
                val_loss += loss.item()
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(ratings.cpu().numpy())
        
        # 保存最后一轮的预测结果
        if epoch == epochs - 1:
            final_preds = all_preds
            final_labels = all_labels
        
        # 计算评估指标
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        mae = mean_absolute_error(all_labels, all_preds)
        val_maes.append(mae)
        
        spearman, _ = spearmanr(all_labels, all_preds)
        val_spearmans.append(spearman)
        
        # 打印评估结果
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, MAE: {mae:.4f}, Spearman: {spearman:.4f}")
    
    # 保存最终模型
    torch.save(model.state_dict(), 'task2/models/simple_model.pth')
    print("模型已保存到 task2/models/simple_model.pth")
    
    # 绘制训练过程
    plot_training_history(train_losses, val_losses, val_maes, val_spearmans, final_labels, final_preds)
    
    return model

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 用户选择
    print("=" * 50)
    print("简易动漫评论评分预测模型微调程序")
    print("=" * 50)
    
    # 询问用户是否使用小数据集
    use_small_dataset = input("是否使用5%的数据集进行快速训练？(y/n): ").lower() == 'y'
    sample_ratio = 0.05 if use_small_dataset else 1.0
    
    # 加载数据
    train_loader, val_loader = load_data(batch_size=32, sample_ratio=sample_ratio)
    
    # 初始化模型
    model = SimpleRatingPredictor('task2/models/bert-base-chinese')
    
    # 训练模型 - 统一使用5轮训练
    epochs = 5
    model = train_and_evaluate(model, train_loader, val_loader, epochs=epochs, device=device)
    
    # 测试预测
    tokenizer = BertTokenizer.from_pretrained('task2/models/bert-base-chinese')
    
    test_comments = [
        "这部动漫太棒了，情节紧凑，人物刻画深入，强烈推荐！",
        "剧情一般，画风还可以，打发时间可以看看。",
        "太难看了，浪费时间，剧情混乱，角色塑造差。"
    ]
    
    print("\n测试预测结果:")
    for comment in test_comments:
        rating = predict_rating(model, comment, tokenizer, device)
        print(f"评论: {comment}")
        print(f"预测评分: {rating:.1f}/10\n")
    
    print("简易微调程序执行完毕!")

def predict_rating(model, text, tokenizer, device='cuda'):
    """使用模型预测单个评论的评分"""
    model.eval()
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        prediction = model(input_ids, attention_mask)
    
    # 将0-1的预测值转换为1-10的评分
    rating = prediction.item() * 10
    return rating

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 用户选择
    print("=" * 50)
    print("简易动漫评论评分预测模型微调程序")
    print("=" * 50)
    
    # 询问用户是否使用小数据集
    use_small_dataset = input("是否使用5%的数据集进行快速训练？(y/n): ").lower() == 'y'
    sample_ratio = 0.05 if use_small_dataset else 1.0
    
    # 加载数据
    train_loader, val_loader = load_data(batch_size=32, sample_ratio=sample_ratio)
    
    # 初始化模型
    model = SimpleRatingPredictor('task2/models/bert-base-chinese')
    
    # 训练模型
    epochs = 3 if use_small_dataset else 5
    model = train_and_evaluate(model, train_loader, val_loader, epochs=epochs, device=device)
    
    # 测试预测
    tokenizer = BertTokenizer.from_pretrained('task2/models/bert-base-chinese')
    
    test_comments = [
        "这部动漫太棒了，情节紧凑，人物刻画深入，强烈推荐！",
        "剧情一般，画风还可以，打发时间可以看看。",
        "太难看了，浪费时间，剧情混乱，角色塑造差。"
    ]
    
    print("\n测试预测结果:")
    for comment in test_comments:
        rating = predict_rating(model, comment, tokenizer, device)
        print(f"评论: {comment}")
        print(f"预测评分: {rating:.1f}/10\n")
    
    print("简易微调程序执行完毕!")