from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

# 检查是否支持 MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def evaluate_model(model_path, test_data_path):
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    # 加载测试数据
    test_df = pd.read_csv(test_data_path)
    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()

    # 数据预处理
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    inputs = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # 模型预测
    with torch.no_grad():
        outputs = model(inputs, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()[:, 1]  # 获取讽刺类别的概率

    # 计算指标
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    # 绘制混淆矩阵
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, 
                annot=True, 
                fmt="d", 
                cmap="Blues", 
                xticklabels=["Non-Sarcastic", "Sarcastic"],  # 英文标签
                yticklabels=["Non-Sarcastic", "Sarcastic"])  # 英文标签
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")  # 保存为图片
    plt.show()

    # 绘制 ROC 曲线
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")  # 保存为图片
    plt.show()

if __name__ == "__main__":
    evaluate_model("model/base/news_model", "data/processed/news_test.csv")