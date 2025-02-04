import os
import torch
import pandas as pd
from transformers import AutoTokenizer
from typing import List, Dict
import nltk
from nltk.corpus import sentiwordnet as swn
from torch.utils.data import DataLoader
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from safetensors.torch import load_file

# 导入训练脚本中的相关类
from train_enhance import (
    KnowledgeBase, 
    EnhancedPreprocessor, 
    KnowledgeEnhancedBERT,
    prepare_knowledge_features,
    DataCollatorForKnowledgeEnhanced
)

def ensure_nltk_resources():
    resources = ['wordnet', 'sentiwordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_model_and_tokenizer(model_path):
    """加载模型和分词器"""
    print(f"Loading model from {model_path}")
    model = KnowledgeEnhancedBERT()
    
    # 使用 safetensors 加载模型权重
    state_dict = load_file(os.path.join(model_path, "model.safetensors"))
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def prepare_test_dataset(df, tokenizer, preprocessor):
    """准备测试数据集"""
    processed_data = []
    for _, row in df.iterrows():
        try:
            text = str(row['text'])
            label = int(row['label'])
            
            # 知识增强预处理
            enhanced_features = preprocessor.preprocess_text(text)
            
            # tokenization
            encoding = tokenizer(
                enhanced_features['processed_text'],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors=None
            )
            
            # 准备知识特征
            knowledge_features = prepare_knowledge_features(enhanced_features).tolist()
            
            processed_data.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'knowledge_features': knowledge_features,
                'labels': label
            })
            
        except Exception as e:
            print(f"Error processing row: {row}")
            print(f"Error message: {str(e)}")
            continue
    
    return Dataset.from_list(processed_data)

def evaluate_model(model, test_dataloader):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            # 将数据移到正确的设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            knowledge_features = batch['knowledge_features'].to(device)
            labels = batch['labels'].to(device)
            
            # 获取预测结果
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                knowledge_features=knowledge_features
            )
            
            # 获取预测的类别
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def calculate_metrics(predictions, labels):
    """计算各种评估指标"""
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    conf_matrix = confusion_matrix(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

def plot_confusion_matrix(conf_matrix, save_path):
    """绘制并保存混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()

def save_results(predictions, labels, metrics, save_dir):
    """保存评估结果"""
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'true_label': labels,
        'predicted_label': predictions
    })
    results_df.to_csv(os.path.join(save_dir, "prediction_results.csv"), index=False)
    
    # 保存指标
    metrics_df = pd.DataFrame([{
        'metric': k,
        'value': v
    } for k, v in metrics.items() if k != 'confusion_matrix'])
    metrics_df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)

def main():
    # 确保NLTK资源可用
    ensure_nltk_resources()
    
    # 加载测试数据
    test_data = pd.read_csv("data/processed/twitter_test.csv")
    
    # 初始化知识库和预处理器
    knowledge_base = KnowledgeBase()
    preprocessor = EnhancedPreprocessor(knowledge_base)
    
    # 加载模型和分词器
    model_path = "model/enhanced/twitter_model"
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # 准备测试数据集
    test_dataset = prepare_test_dataset(test_data, tokenizer, preprocessor)
    
    # 创建数据加载器
    data_collator = DataCollatorForKnowledgeEnhanced(tokenizer)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        collate_fn=data_collator,
        shuffle=False
    )
    
    # 评估模型
    print("Evaluating model...")
    predictions, labels = evaluate_model(model, test_dataloader)
    
    # 计算指标
    metrics = calculate_metrics(predictions, labels)
    
    # 打印结果
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # 创建实验结果目录
    results_dir = "experiments/enhanced_model_evaluation"
    os.makedirs(results_dir, exist_ok=True)
    
    # 绘制并保存混淆矩阵
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        os.path.join(results_dir, "confusion_matrix.png")
    )
    
    # 保存详细结果
    save_results(predictions, labels, metrics, results_dir)
    
    print(f"\nResults have been saved to {results_dir}")
    print(f"Confusion matrix has been saved to {os.path.join(results_dir, 'confusion_matrix.png')}")

if __name__ == "__main__":
    main()
