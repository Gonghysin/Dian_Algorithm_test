import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig
import os
import torch.nn as nn
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from scipy.stats import spearmanr
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json

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

def load_data(batch_size=16):
    """加载数据集"""
    print("开始加载数据...")
    
    # 加载训练集和验证集
    train_df = pd.read_csv('task2/dataset/train_set.csv')
    val_df = pd.read_csv('task2/dataset/val_set.csv')
    
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('task2/models/bert-base-chinese')
    
    # 创建数据集实例
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
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, tokenizer

class AnimeRatingPredictor(nn.Module):
    def __init__(self, pretrained_model_path):
        super().__init__()
        config = BertConfig.from_json_file(f'{pretrained_model_path}/config.json')
        self.bert = BertModel.from_pretrained(
            pretrained_model_path,
            config=config,
            local_files_only=True,
            use_safetensors=False,
            from_tf=False,
            _fast_init=True
        )
        
        # 冻结部分BERT参数
        for param in list(self.bert.parameters())[:-6]:
            param.requires_grad = False
            
        # 使用双输出头，一个预测评分范围，一个预测具体评分
        self.shared_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.4)
        )
        
        # 评分区间分类器(0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
        self.classifier = nn.Linear(256, 5)
        
        # 具体评分回归器
        self.regressor = nn.Linear(256, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        shared_features = self.shared_layers(cls_output)
        
        # 区间分类预测
        logits = self.classifier(shared_features)
        
        # 回归预测 - 扩大范围到[-0.2, 1.2]
        regression = self.regressor(shared_features).squeeze(-1)
        
        # 使用Tanh但扩大范围为[-0.2, 1.2]
        regression = 0.7 * (torch.tanh(regression) + 1.0) - 0.2
        
        # 在训练阶段保持扩展范围，不裁剪
        # 在评估/预测时再裁剪到[0,1]范围
        if not self.training:
            regression = torch.clamp(regression, 0.0, 1.0)
        
        return regression, logits

# 组合损失函数
class CombinedLoss(nn.Module):
    def __init__(self, class_weight=0.3, focal_weight=2.0, gamma=1.0):
        super().__init__()
        self.class_weight = class_weight
        self.focal_weight = focal_weight
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, regression_pred, class_pred, target):
        # 计算MSE前先对预测值进行裁剪，避免负面惩罚超出范围的预测
        clipped_pred = torch.clamp(regression_pred, 0.0, 1.0)
        
        # 回归损失（MSE）
        mse = (clipped_pred - target) ** 2
        
        # 对低评分样本加权
        weights = torch.ones_like(target)
        weights[target <= 0.3] = self.focal_weight
        
        # Focal loss调整
        difficult_factor = torch.exp(self.gamma * mse)
        
        # 分类目标
        target_class = torch.zeros_like(target, dtype=torch.long)
        target_class[(target >= 0.0) & (target < 0.2)] = 0
        target_class[(target >= 0.2) & (target < 0.4)] = 1
        target_class[(target >= 0.4) & (target < 0.6)] = 2
        target_class[(target >= 0.6) & (target < 0.8)] = 3
        target_class[(target >= 0.8) & (target <= 1.0)] = 4
        
        # 分类损失
        cls_loss = self.ce_loss(class_pred, target_class)
        
        # 组合损失
        reg_loss = (weights * difficult_factor * mse).mean()
        combined_loss = reg_loss + self.class_weight * cls_loss
        
        return combined_loss, reg_loss, cls_loss

def evaluate_model(model, data_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ratings = batch['rating'].to(device)
            
            with autocast():
                regression, logits = model(input_ids, attention_mask)
                # 使用组合损失函数
                combined_loss, reg_loss, cls_loss = criterion(regression, logits, ratings)
            
            total_loss += combined_loss.item()
            all_preds.extend(regression.cpu().numpy())
            all_labels.extend(ratings.cpu().numpy())
    
    # 计算各种评估指标
    mse = total_loss / len(data_loader)
    mae = mean_absolute_error(all_labels, all_preds)
    spearman_corr, _ = spearmanr(all_labels, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_preds)
    
    # 计算低分段（1-3分）的预测效果
    low_ratings_mask = np.array(all_labels) <= 0.3
    low_ratings_mse = mean_squared_error(
        np.array(all_labels)[low_ratings_mask],
        np.array(all_preds)[low_ratings_mask]
    ) if any(low_ratings_mask) else float('inf')
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'spearman': spearman_corr,
        'low_ratings_mse': low_ratings_mse
    }

def train_model(model, train_loader, val_loader, epochs=15, device='cuda',
                patience=5, weight_decay=0.01):
    model = model.to(device)
    
    # 使用组合损失函数
    criterion = CombinedLoss(class_weight=0.3, focal_weight=2.0, gamma=1.0)
    
    # 分层学习率
    optimizer = AdamW([
        {"params": model.bert.parameters(), "lr": 2e-5},
        {"params": model.shared_layers.parameters(), "lr": 5e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
        {"params": model.regressor.parameters(), "lr": 1e-3}
    ], weight_decay=weight_decay)
    
    # 学习率调度
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[2e-5, 5e-4, 1e-3, 1e-3],
        total_steps=total_steps,
        pct_start=0.1
    )
    
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    print("开始训练...")
    start_time = time.time()
    
    # 修改记录训练历史的指标名称，使其与evaluate_model返回的指标匹配
    metrics_history = {
        'train_mse': [], 'val_mse': [],
        'train_rmse': [], 'val_rmse': [],
        'train_mae': [], 'val_mae': [],
        'train_r2': [], 'val_r2': [],
        'train_spearman': [], 'val_spearman': [],
        'train_low_ratings_mse': [], 'val_low_ratings_mse': []
    }
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ratings = batch['rating'].to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                regression, logits = model(input_ids, attention_mask)
                combined_loss, reg_loss, cls_loss = criterion(regression, logits, ratings)
            
            scaler.scale(combined_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_train_loss += combined_loss.item()
            train_pbar.set_postfix({'loss': f'{combined_loss.item():.4f}'})
        
        # 收集验证集的预测结果
        model.eval()
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                ratings = batch['rating'].to(device)
                regression, _ = model(input_ids, attention_mask)
                val_preds.extend(regression.cpu().numpy())
                val_labels.extend(ratings.cpu().numpy())
        
        # 评估阶段
        train_metrics = evaluate_model(model, train_loader, criterion, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        # 打印详细的评估指标
        print(f'\nEpoch {epoch+1}:')
        print(f'Training - MSE: {train_metrics["mse"]:.4f}, MAE: {train_metrics["mae"]:.4f}, '
              f'Spearman: {train_metrics["spearman"]:.4f}, Low Ratings MSE: {train_metrics["low_ratings_mse"]:.4f}')
        print(f'Validation - MSE: {val_metrics["mse"]:.4f}, MAE: {val_metrics["mae"]:.4f}, '
              f'Spearman: {val_metrics["spearman"]:.4f}, Low Ratings MSE: {val_metrics["low_ratings_mse"]:.4f}')
        
        # 早停检查
        if val_metrics['mse'] < best_val_loss:
            best_val_loss = val_metrics['mse']
            patience_counter = 0
            best_epoch = epoch
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
            }, 'task2/models/best_model.pth')
            print(f'Best model saved with validation MSE: {best_val_loss:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after epoch {epoch+1}')
                break
        
        # 记录指标
        for metric in ['mse', 'rmse', 'mae', 'r2', 'spearman', 'low_ratings_mse']:
            metrics_history[f'train_{metric}'].append(train_metrics[metric])
            metrics_history[f'val_{metric}'].append(val_metrics[metric])
        
        # 每个epoch结束时绘制图表，使用收集到的验证集结果
        plot_training_metrics(metrics_history, all_labels=val_labels, all_preds=val_preds)
        
        # 在验证集上生成混淆矩阵（使用当前epoch的预测结果）
        if epoch == best_epoch:
            plot_confusion_matrix(
                np.array(val_labels), 
                np.array(val_preds),
                'task2/models/confusion_matrix.png'
            )
    
    training_time = time.time() - start_time
    print(f'\nTraining completed in {training_time/60:.2f} minutes')
    print(f'Best model was saved at epoch {best_epoch+1}')
    
    # 训练结束后保存完整的评估结果
    final_results = {
        'metrics_history': metrics_history,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'training_time': training_time,
        'final_train_metrics': train_metrics,
        'final_val_metrics': val_metrics
    }
    
    with open('task2/models/training_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)

def plot_training_metrics(metrics_history, all_labels=None, all_preds=None):
    """绘制训练过程中的指标变化"""
    plt.figure(figsize=(15, 10))
    
    # 1. 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(metrics_history['train_mse'], label='Train MSE')
    plt.plot(metrics_history['val_mse'], label='Val MSE')
    plt.title('MSE Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    # 2. MAE曲线
    plt.subplot(2, 2, 2)
    plt.plot(metrics_history['train_mae'], label='Train MAE')
    plt.plot(metrics_history['val_mae'], label='Val MAE')
    plt.title('MAE over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
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
    plt.savefig('task2/models/training_metrics.png')
    plt.close()

def plot_confusion_matrix(true_ratings, pred_ratings, save_path):
    """绘制评分区间混淆矩阵"""
    # 将连续值转换为分类区间
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['1-2', '3-4', '5-6', '7-8', '9-10']
    
    # 确保输入是numpy数组
    true_ratings = np.array(true_ratings)
    pred_ratings = np.array(pred_ratings)
    
    # 使用numpy的digitize函数进行分箱
    true_bins = np.digitize(true_ratings, bins) - 1
    pred_bins = np.digitize(pred_ratings, bins) - 1
    
    # 将超出范围的值限制在合法范围内
    true_bins = np.clip(true_bins, 0, len(labels)-1)
    pred_bins = np.clip(pred_bins, 0, len(labels)-1)
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_bins, pred_bins, labels=range(len(labels)))
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title('Rating Range Confusion Matrix')
    plt.xlabel('Predicted Rating')
    plt.ylabel('True Rating')
    plt.savefig(save_path)
    plt.close()

def test_tokenization():
    """测试分词效果"""
    tokenizer = BertTokenizer.from_pretrained('task2/models/bert-base-chinese')
    
    # 测试样例
    sample_text = "白开水"
    tokens = tokenizer.tokenize(sample_text)
    token_ids = tokenizer.encode(sample_text)
    
    print(f"原文: {sample_text}")
    print(f"分词结果: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"解码结果: {tokenizer.decode(token_ids)}")

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备和batch size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    
    # 加载数据
    train_loader, val_loader, tokenizer = load_data(batch_size=batch_size)
    
    # 初始化模型
    model = AnimeRatingPredictor('task2/models/bert-base-chinese')
    
    # 训练模型
    train_model(
        model,
        train_loader,
        val_loader,
        epochs=15,
        device=device,
        patience=5,
        weight_decay=0.01
    )
    test_tokenization()  # 添加这行
