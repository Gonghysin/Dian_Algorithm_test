import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import pickle
from tqdm import tqdm  # 添加进度条支持
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据集
def load_data(file_path):
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    data = pd.read_csv(file_path, header=None, names=column_names)
    return data

# 划分数据集
def split_dataset(data):
    X = data.drop('species', axis=1)
    y = data['species']
    
    # 按照7:3的比例划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3,   
        random_state=42,  
        stratify=y        # 保证划分后类别比例一致
    )
    
    return X_train, X_test, y_train, y_test

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # 分裂特征
        self.threshold = threshold  # 分裂阈值
        self.left = left           # 左子树
        self.right = right         # 右子树
        self.value = value         # 叶节点的预测值

class DecisionTree: # 决策树类
    def __init__(self, max_depth=5, min_samples=2): # 最大深度为5，最小样本数为2
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None
        self.label_encoder = {}  # 添加标签编码器
        self.label_decoder = {}  # 添加标签解码器

    def encode_labels(self, y):
        """将字符串标签编码为数值"""
        if not self.label_encoder:
            unique_labels = np.unique(y)
            self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
            self.label_decoder = {i: label for i, label in enumerate(unique_labels)}
        return np.array([self.label_encoder[label] for label in y])

    def decode_labels(self, y):
        """将数值标签解码为字符串"""
        return np.array([self.label_decoder[i] for i in y])

    def calculate_gini(self, y):
        """计算基尼指数"""
        unique_classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def find_best_split(self, X, y):
        """找到最佳分裂特征和阈值"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                gini_left = self.calculate_gini(y[left_mask])
                gini_right = self.calculate_gini(y[right_mask])

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                gini = (n_left * gini_left + n_right * gini_right) / n_samples

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gini

    def build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        n_samples = len(y)
        
        # 编码标签
        y_encoded = self.encode_labels(y)
        

        # 检查停止条件
        if len(np.unique(y_encoded)) == 1:
            return Node(value=y[0])  # 返回原始标签
        
        # 2. 样本数量小于最小值
        if n_samples < self.min_samples:
            most_common_idx = np.bincount(y_encoded).argmax()
            return Node(value=self.label_decoder[most_common_idx])
        
        # 3. 树深度达到最大值
        if depth >= self.max_depth:
            most_common_idx = np.bincount(y_encoded).argmax()
            return Node(value=self.label_decoder[most_common_idx])

        # 寻找最佳分裂点
        feature, threshold, gini = self.find_best_split(X, y_encoded)
        
        # 如果找不到有效的分裂点
        if feature is None:
            most_common_idx = np.bincount(y_encoded).argmax()
            return Node(value=self.label_decoder[most_common_idx])

        # 根据最佳分裂点划分数据
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # 递归构建左右子树
        left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def fit(self, X, y):
        """训练决策树"""
        self.tree = self.build_tree(X, y)
        return self

    def predict_one(self, x):
        """预测单个样本"""
        node = self.tree
        while node.value is None:  # 非叶节点
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
    
    def predict(self, X):
        """预测多个样本"""
        return np.array([self.predict_one(x) for x in X])

    def set_feature_selector(self, selector_function):
        """设置特征选择器函数"""
        self.find_best_split = selector_function

class RandomForest:
    def __init__(self, n_trees=100, max_depth=5, min_samples=2, n_features=2):
        """
        初始化随机森林
        n_trees: 决策树数量，建议100棵
        max_depth: 每棵树的最大深度，默认5
        min_samples: 节点最小样本数，默认2
        n_features: 特征子集大小，建议2个特征
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_features = n_features
        self.trees = []
        # 设置默认特征名称
        self.feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    def bootstrap_sample(self, X, y):
        """Bootstrap采样：有放回地随机抽取样本"""
        n_samples = X.shape[0]
        # 随机抽取n_samples个样本的索引（有放回）
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def find_best_split_with_feature_subset(self, tree, X, y):
        """在特征子集上寻找最佳分裂点"""
        n_features = X.shape[1]
        feature_subset = np.random.choice(
            n_features, 
            size=self.n_features, 
            replace=False
        )
        
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature in feature_subset:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                    
                gini_left = tree.calculate_gini(y[left_mask])
                gini_right = tree.calculate_gini(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                gini = (n_left * gini_left + n_right * gini_right) / len(y)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gini

    def fit(self, X, y):
        """训练随机森林"""
        print(f"\n=== 开始训练随机森林 ===")
        print(f"配置信息:")
        print(f"- 决策树数量: {self.n_trees}")
        print(f"- 最大树深度: {self.max_depth}")
        print(f"- 最小样本数: {self.min_samples}")
        print(f"- 特征子集大小: {self.n_features}")
        
        # 如果输入是DataFrame，使用其列名
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # 构建多棵决策树
        for i in tqdm(range(self.n_trees), desc="训练决策树"):
            # Bootstrap采样
            X_sample, y_sample = self.bootstrap_sample(X, y)
            
            # 创建决策树
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples=self.min_samples
            )
            
            # 绑定特征选择方法
            tree.original_find_best_split = tree.find_best_split  # 保存原始方法
            tree.set_feature_selector(
                lambda X, y: self.find_best_split_with_feature_subset(tree, X, y)
            )
            
            # 训练决策树
            tree.fit(X_sample, y_sample)
            
            # 恢复原始方法（为了能够序列化）
            tree.find_best_split = tree.original_find_best_split
            del tree.original_find_best_split
            
            # 将训练好的树添加到森林中
            self.trees.append(tree)
            
        print("\n=== 随机森林训练完成 ===")
        return self

    def save_model(self, filepath):
        """保存随机森林模型"""
        model_data = {
            'trees': self.trees,
            'n_trees': self.n_trees,
            'max_depth': self.max_depth,
            'min_samples': self.min_samples,
            'n_features': self.n_features,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n模型已保存到: {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """加载随机森林模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        forest = cls(
            n_trees=model_data['n_trees'],
            max_depth=model_data['max_depth'],
            min_samples=model_data['min_samples'],
            n_features=model_data['n_features']
        )
        forest.trees = model_data['trees']
        forest.feature_names = model_data['feature_names']
        print(f"\n模型已从 {filepath} 加载")
        return forest

    def predict_one_with_proba(self, x):
        """预测单个样本（包含投票情况）"""
        # 收集所有树的投票
        predictions = []
        for tree in self.trees:
            pred = tree.predict_one(x)
            predictions.append(pred)
        
        # 统计投票结果
        unique_labels, counts = np.unique(predictions, return_counts=True)
        proba = counts / len(predictions)
        
        # 返回预测结果和概率
        winner_idx = np.argmax(counts)
        return unique_labels[winner_idx], dict(zip(unique_labels, proba))

    def predict(self, X):
        """预测多个样本"""
        predictions = []
        probabilities = []
        
        print("\n开始预测...")
        for i, x in enumerate(X):
            pred, proba = self.predict_one_with_proba(x)
            predictions.append(pred)
            probabilities.append(proba)
            
        return np.array(predictions), probabilities

    def evaluate(self, X, y_true):
        """评估模型性能"""
        y_pred, probabilities = self.predict(X)
        
        # 1. 计算准确率
        accuracy = np.mean(y_pred == y_true)
        
        # 2. 构建混淆矩阵
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(unique_labels)
        confusion_matrix = np.zeros((n_classes, n_classes))
        
        for i in range(len(y_true)):
            true_idx = np.where(unique_labels == y_true[i])[0][0]
            pred_idx = np.where(unique_labels == y_pred[i])[0][0]
            confusion_matrix[true_idx][pred_idx] += 1
            
        # 3. 计算每个类别的精确率、召回率和F1分数
        metrics = {}
        for i, label in enumerate(unique_labels):
            # 真阳性(TP)、假阳性(FP)、假阴性(FN)
            tp = confusion_matrix[i, i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp
            
            # 计算精确率
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            # 计算召回率
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            # 计算F1分数
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix,
            'class_metrics': metrics,
            'labels': unique_labels
        }

    def feature_importance(self, X, y):
        """
        计算特征重要性
        X: 用于计算特征重要性的数据
        y: 对应的标签
        """
        importances = np.zeros(len(self.feature_names))
        
        # 收集所有树中每个特征的基尼指数减少量
        for tree in self.trees:
            def collect_gini_decrease(node, X_node, y_node):
                if node.value is not None:  # 叶节点
                    return
                
                # 计算当前节点的基尼指数减少量
                feature = node.feature
                
                # 获取左右子节点的样本比例和基尼指数
                left_mask = X_node[:, feature] <= node.threshold
                right_mask = ~left_mask
                
                # 如果划分后的节点为空，则跳过
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    return
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                n_total = n_left + n_right
                
                # 计算分裂前的基尼指数
                gini_parent = tree.calculate_gini(y_node)
                
                # 计算分裂后的加权基尼指数
                gini_left = tree.calculate_gini(y_node[left_mask])
                gini_right = tree.calculate_gini(y_node[right_mask])
                gini_children = (n_left * gini_left + n_right * gini_right) / n_total
                
                # 计算基尼指数减少量
                gini_decrease = gini_parent - gini_children
                importances[feature] += gini_decrease
                
                # 递归遍历子节点
                collect_gini_decrease(node.left, X_node[left_mask], y_node[left_mask])
                collect_gini_decrease(node.right, X_node[right_mask], y_node[right_mask])
            
            collect_gini_decrease(tree.tree, X, y)
        
        # 计算平均重要性并归一化
        importances = importances / self.n_trees
        if np.sum(importances) > 0:  # 避免除以0
            importances = importances / np.sum(importances)
        
        # 创建特征重要性字典
        importance_dict = dict(zip(self.feature_names, importances))
        
        # 按重要性降序排序
        sorted_importance = sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_importance

def plot_confusion_matrix(confusion_matrix, labels):
    """绘制混淆矩阵热力图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()

def plot_feature_importance(feature_importance):
    """绘制特征重要性条形图"""
    features, importances = zip(*feature_importance)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(features, importances)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.title('Feature Importance in Random Forest')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 加载数据
    data = load_data('task1/iris_dataset/bezdekIris.data')
    
    # 划分数据集
    X_train, X_test, y_train, y_test = split_dataset(data)
    
    # 打印数据集大小,验证划分结果
    print("训练集大小:", X_train.shape)
    print("测试集大小:", X_test.shape)
    
    # 训练随机森林
    print("\n=== 开始随机森林训练 ===")
    rf = RandomForest(n_trees=100, max_depth=5, min_samples=2, n_features=2)
    rf.fit(X_train.values, y_train.values)
    
    # 计算并显示特征重要性
    print("\n=== 特征重要性分析 ===")
    feature_importance = rf.feature_importance(X_train.values, y_train.values)
    print("\n特征重要性排序:")
    for feature, importance in feature_importance:
        print(f"{feature:12}: {importance:.4f}")
    
    # 绘制特征重要性图
    plot_feature_importance(feature_importance)
    
    # 保存模型
    rf.save_model('task1/random_forest_model.pkl')

    # 加载保存的模型
    rf = RandomForest.load_model('task1/random_forest_model.pkl')
    
    # 单个样本预测示例
    print("\n=== 单个样本预测示例 ===")
    sample_idx = 0
    sample = X_test.iloc[sample_idx]
    true_label = y_test.iloc[sample_idx]
    
    pred, proba = rf.predict_one_with_proba(sample.values)
    print(f"样本特征: {dict(zip(X_test.columns, sample.values))}")
    print(f"真实类别: {true_label}")
    print(f"预测类别: {pred}")
    print(f"预测概率: {proba}")
    
    # 整个测试集的预测
    print("\n=== 测试集预测 ===")
    predictions, probabilities = rf.predict(X_test.values)
    
    # 计算准确率
    accuracy = np.mean(predictions == y_test.values)
    print(f"\n测试集准确率: {accuracy:.4f}")
    
    # 显示部分预测结果
    print("\n部分预测结果示例:")
    for i in range(min(5, len(X_test))):
        print(f"\n样本 {i+1}:")
        print(f"真实类别: {y_test.iloc[i]}")
        print(f"预测类别: {predictions[i]}")
        print(f"预测概率: {probabilities[i]}")

    # 模型评估
    print("\n=== 模型评估 ===")
    
    # 1. 在测试集上评估
    print("\n在测试集上的评估结果:")
    eval_results = rf.evaluate(X_test.values, y_test.values)
    
    print(f"\n整体准确率: {eval_results['accuracy']:.4f}")
    
    print("\n各类别的评估指标:")
    for label in eval_results['labels']:
        metrics = eval_results['class_metrics'][label]
        print(f"\n类别 {label}:")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")
    
    # 绘制混淆矩阵
    plot_confusion_matrix(eval_results['confusion_matrix'], eval_results['labels'])
    
    # 2. 5折交叉验证
    print("\n=== 5折交叉验证 ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies = []
    
    # 合并训练集和测试集进行交叉验证
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full), 1):
        # 获取当前折的训练集和验证集
        X_fold_train = X_full.iloc[train_idx].values
        y_fold_train = y_full.iloc[train_idx].values
        X_fold_val = X_full.iloc[val_idx].values
        y_fold_val = y_full.iloc[val_idx].values
        
        # 训练模型
        print(f"\n训练第 {fold} 折...")
        rf = RandomForest(n_trees=100, max_depth=5, min_samples=2, n_features=2)
        rf.fit(X_fold_train, y_fold_train)
        
        # 评估当前折
        fold_results = rf.evaluate(X_fold_val, y_fold_val)
        cv_accuracies.append(fold_results['accuracy'])
        print(f"第 {fold} 折准确率: {fold_results['accuracy']:.4f}")
    
    # 输出交叉验证结果
    print("\n交叉验证结果:")
    print(f"平均准确率: {np.mean(cv_accuracies):.4f}")
    print(f"标准差: {np.std(cv_accuracies):.4f}")
