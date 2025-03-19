# Dian_Algorithm_test
Dian团队2025春招算法测试题

## 环境配置

创建conda Python 3.11.11虚拟环境

```bash
conda create -n dian python=3.11.11
```

激活虚拟环境

```bash
conda activate dian
```

安装依赖

```bash
pip install -r requirements.txt
```

# 任务完成情况

## Task1 随机森林的理解与实现-鸢尾花分类

详细完成路径，和输出结果见：[task1_doc.md](./task1/task1_doc.md)

使用方法：运行./task1/Random_forest.py


## Task2 Bangumi评论分数预测器的训练

详细完成路径，和输出结果见：[task2_doc.md](./task2/task2_doc.md)

项目文件结构：

```bash
task2/
├── catch/ # 爬虫脚本
├── dataset/ # 数据集
├── dataset_proceed.py # 数据集处理脚本
├── only_highdata_image.py # 数据集可视化脚本（单独用来可视化原数据集用的）
├── task2_doc.md # 任务文档
├── train_model.py # 训练模型脚本
└── dataset_clear.py # 数据集清洗脚本
```


模型已上传到HuggingFace
地址：https://huggingface.co/titicacine/chinese-text-rating-model
内附有使用方法

## Task3 注意⼒机制及其变体的理解与实现

详细完成路径见:[任务三完成结果](./task3/task3_doc.md)