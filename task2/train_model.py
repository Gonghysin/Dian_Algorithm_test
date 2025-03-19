import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig, AdamW
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertConfig, AdamW
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os
from huggingface_model.inference import predict_rating

# 确保模型保存目录存在
os.makedirs('task2/models', exist_ok=True)

def load_data(batch_size=32, sample_ratio=1.0):
    """加载训练和验证数据"""
    tokenizer = BertTokenizer.from_pretrained('task2/models/bert-base-chinese')
    train_df = pd.read_csv('task2/dataset/train_set.csv')
    val_df = pd.read_csv('task2/dataset/val_set.csv')
    
    # 根据sample_ratio采样数据
    if sample_ratio < 1.0:
        train_df = train_df.sample(frac=sample_ratio, random_state=42)
        val_df = val_df.sample(frac=sample_ratio, random_state=42)
    
    train_dataset = AnimeDataset(train_df['comment'].tolist(), train_df['normalized_rating'].tolist(), tokenizer)
    val_dataset = AnimeDataset(val_df['comment'].tolist(), val_df['normalized_rating'].tolist(), tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

class AnimeDataset(Dataset):
    """简化的数据集类"""
    def __init__(self, comments, ratings, tokenizer, max_length=128):
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
    """简单的评分预测模型"""
    def __init__(self, pretrained_model_path, dropout_rate=0.3):
        super().__init__()
        config = BertConfig.from_json_file(f'{pretrained_model_path}/config.json')
        self.bert = BertModel.from_pretrained(
            pretrained_model_path,
            config=config,
            local_files_only=True
        )
        
        # 简单的回归头
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.regressor(cls_output).squeeze(-1)

class WeightedMSELoss(nn.Module):
    """加权MSE损失函数，对低评分样本给予更高权重"""
    def __init__(self, low_rating_weight=2.0, low_rating_threshold=0.3):
        super().__init__()
        self.low_rating_weight = low_rating_weight
        self.low_rating_threshold = low_rating_threshold
        
    def forward(self, predictions, targets):
        # 计算平方误差
        squared_errors = (predictions - targets) ** 2
        
        # 为低评分样本分配更高权重
        weights = torch.ones_like(targets)
        weights[targets <= self.low_rating_threshold] = self.low_rating_weight
        
        # 计算加权平均
        weighted_loss = (weights * squared_errors).mean()
        return weighted_loss

def train_and_evaluate(model, train_loader, val_loader, epochs=15, lr=2e-5, 
                       patience=3, weight_decay=0.01, low_rating_weight=2.0, device='cuda'):
    """训练和评估模型，添加早停策略和学习率调整"""
    model = model.to(device)
    
    # 使用加权MSE损失函数
    criterion = WeightedMSELoss(low_rating_weight=low_rating_weight)
    
    # 添加权重衰减以增强正则化
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度器 - 当验证损失不再下降时降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    val_maes = []
    val_spearmans = []
    val_mses = []
    val_rmses = []
    val_low_rating_mses = []  # 添加低评分样本MSE记录
    
    # 记录最后一轮的预测结果
    final_preds = None
    final_labels = None
    
    # 早停变量
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    best_epoch = 0
    
    print("开始训练...")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
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
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
            total_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 评估阶段
        # 评估阶段
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
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
        
        # 保存当前轮次的预测结果
        current_preds = np.array(all_preds)
        current_labels = np.array(all_labels)
        
        # 计算评估指标
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 更新学习率调度器
        scheduler.step(avg_val_loss)
        
        # 计算MSE和RMSE
        mse = mean_squared_error(current_labels, current_preds)
        rmse = np.sqrt(mse)
        val_mses.append(mse)
        val_rmses.append(rmse)
        
        # 计算低评分样本的MSE
        low_rating_mask = current_labels <= 0.3
        if np.any(low_rating_mask):
            low_rating_mse = mean_squared_error(
                current_labels[low_rating_mask], 
                current_preds[low_rating_mask]
            )
        else:
            low_rating_mse = float('nan')
        val_low_rating_mses.append(low_rating_mse)
        
        mae = mean_absolute_error(current_labels, current_preds)
        val_maes.append(mae)
        
        spearman, _ = spearmanr(current_labels, current_preds)
        val_spearmans.append(spearman)
        
        # 打印评估结果
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, Spearman: {spearman:.4f}")
        print(f"低评分样本MSE: {low_rating_mse:.4f}")
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            final_preds = current_preds
            final_labels = current_labels
            print(f"发现更好的模型! 验证损失: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            print(f"验证损失未改善. 早停计数器: {early_stop_counter}/{patience}")
            
            if early_stop_counter >= patience:
                print(f"\n早停触发! 在第{epoch+1}轮停止训练.")
                print(f"最佳模型出现在第{best_epoch+1}轮，验证损失为{best_val_loss:.4f}")
                break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"已恢复第{best_epoch+1}轮的最佳模型")
    
    # 保存最终模型
    torch.save(model.state_dict(), 'task2/models/simple_model.pth')
    print("模型已保存到 task2/models/simple_model.pth")
    
    # 绘制训练过程
    plot_training_history(
        train_losses, val_losses, val_maes, val_spearmans, 
        val_mses, val_rmses, val_low_rating_mses, final_labels, final_preds,
        best_epoch
    )
    
    # 打印最终评估结果
    print("\n最终评估结果 (最佳模型):")
    print(f"MSE: {val_mses[best_epoch]:.4f}")
    print(f"RMSE: {val_rmses[best_epoch]:.4f}")
    print(f"MAE: {val_maes[best_epoch]:.4f}")
    print(f"Spearman相关系数: {val_spearmans[best_epoch]:.4f}")
    print(f"低评分样本MSE: {val_low_rating_mses[best_epoch]:.4f}")
    
    return model

def plot_training_history(train_losses, val_losses, val_maes, val_spearmans, 
                          val_mses, val_rmses, val_low_rating_mses, 
                          all_labels=None, all_preds=None, best_epoch=None):
    """绘制更详细的训练历史图表，包括低评分样本MSE和最佳模型标记"""
    plt.figure(figsize=(15, 12))
    
    # 1. 损失曲线
    plt.subplot(3, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    if best_epoch is not None:
        plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Model')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 2. MSE曲线
    plt.subplot(3, 2, 2)
    plt.plot(val_mses, label='MSE')
    plt.plot(val_low_rating_mses, label='Low Rating MSE')
    if best_epoch is not None:
        plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Model')
    plt.title('MSE over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/MSE')
    plt.legend()
    
    # 3. MAE和RMSE曲线
    plt.subplot(3, 2, 3)
    plt.plot(val_maes, label='MAE')
    plt.plot(val_rmses, label='RMSE')
    if best_epoch is not None:
        plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Model')
    plt.title('MAE and RMSE over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    
    # 4. Spearman相关系数曲线
    plt.subplot(3, 2, 4)
    plt.plot(val_spearmans, label='Spearman')
    if best_epoch is not None:
        plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Model')
    plt.title('Spearman Correlation over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    plt.legend()
    
    # 5. 预测值vs真实值散点图
    if all_labels is not None and all_preds is not None:
        plt.subplot(3, 2, 5)
        plt.scatter(all_labels, all_preds, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        # 高亮低评分样本
        low_rating_mask = all_labels <= 0.3
        if np.any(low_rating_mask):
            plt.scatter(
                np.array(all_labels)[low_rating_mask], 
                np.array(all_preds)[low_rating_mask], 
                color='red', alpha=0.7, label='Low Ratings'
            )
        plt.title('Predictions vs Ground Truth')
        plt.xlabel('True Ratings')
        plt.ylabel('Predicted Ratings')
        plt.legend()
    
    # 6. 评分分布对比
    if all_labels is not None and all_preds is not None:
        plt.subplot(3, 2, 6)
        plt.hist(all_labels, bins=20, alpha=0.5, label='True')
        plt.hist(all_preds, bins=20, alpha=0.5, label='Predicted')
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('task2/models/simple_training_history.png')
    plt.savefig('task2/models/simple_training_history.png')
    plt.close()
    print("训练历史图表已保存到 task2/models/simple_training_history.png")

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
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
    print(f"使用设备: {device}")
    
    # 用户选择
    print("=" * 50)
    print("简易动漫评论评分预测模型微调程序")
    print("=" * 50)
    
    # 询问用户是否使用小数据集
    use_small_dataset = input("是否使用5%的数据集进行快速训练？(y/n): ").lower() == 'y'
    sample_ratio = 0.05 if use_small_dataset else 1.0
    
    # 加载数据
    train_loader, val_loader = load_data(batch_size=64, sample_ratio=sample_ratio)
    
    # 初始化模型 - 增加dropout率
    model = SimpleRatingPredictor('task2/models/bert-base-chinese', dropout_rate=0.3)
    
    # 训练模型 - 增加轮次，添加早停和学习率调整
    epochs = 15  # 增加到15轮
    patience = 3  # 早停耐心值
    weight_decay = 0.01  # L2正则化
    low_rating_weight = 2.0  # 低评分样本权重
    
    model = train_and_evaluate(
        model, 
        train_loader, 
        val_loader, 
        epochs=epochs, 
        patience=patience,
        weight_decay=weight_decay,
        low_rating_weight=low_rating_weight,
        device=device
    )
    
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
