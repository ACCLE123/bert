from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import BertTokenizer
from train_multi import MultiTaskBertModel  # 导入我们的多任务模型
import pandas as pd
import numpy as np
from safetensors.torch import load_file

# 检查是否支持 MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def evaluate_model(model_path, test_data_path):
    # 加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = MultiTaskBertModel("bert-base-uncased", num_labels_task1=2, num_labels_task2=2)
    state_dict = load_file(f"{model_path}/model.safetensors")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 加载测试数据
    test_df = pd.read_csv(test_data_path)
    texts = test_df["text"].tolist()
    labels_task1 = test_df["label"].tolist()
    labels_task2 = test_df["sentiment_label"].tolist()

    # 数据预处理
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    inputs = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # 模型预测
    with torch.no_grad():
        logits_task1, logits_task2 = model(inputs, attention_mask=attention_mask)
        predictions_task1 = torch.argmax(logits_task1, dim=1).cpu().numpy()
        predictions_task2 = torch.argmax(logits_task2, dim=1).cpu().numpy()
        probs_task1 = torch.softmax(logits_task1, dim=1).cpu().numpy()[:, 1]
        probs_task2 = torch.softmax(logits_task2, dim=1).cpu().numpy()[:, 1]

    # 评估任务1（标签预测）
    print("\nTask 1 (Label Prediction) Results:")
    accuracy1 = accuracy_score(labels_task1, predictions_task1)
    f1_1 = f1_score(labels_task1, predictions_task1)
    print(f"Accuracy: {accuracy1:.4f}, F1 Score: {f1_1:.4f}")

    # 评估任务2（情感预测）
    print("\nTask 2 (Sentiment Prediction) Results:")
    accuracy2 = accuracy_score(labels_task2, predictions_task2)
    f1_2 = f1_score(labels_task2, predictions_task2)
    print(f"Accuracy: {accuracy2:.4f}, F1 Score: {f1_2:.4f}")

    # 绘制任务1的混淆矩阵
    plt.figure(figsize=(6, 6))
    cm1 = confusion_matrix(labels_task1, predictions_task1)
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Class 0", "Class 1"],
                yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Task 1 - Confusion Matrix")
    plt.savefig("confusion_matrix_task1.png")
    plt.close()

    # 绘制任务2的混淆矩阵
    plt.figure(figsize=(6, 6))
    cm2 = confusion_matrix(labels_task2, predictions_task2)
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Class 0", "Class 1"],
                yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Task 2 - Confusion Matrix")
    plt.savefig("confusion_matrix_task2.png")
    plt.close()

    # 绘制任务1的ROC曲线
    plt.figure(figsize=(8, 6))
    fpr1, tpr1, _ = roc_curve(labels_task1, probs_task1)
    roc_auc1 = auc(fpr1, tpr1)
    plt.plot(fpr1, tpr1, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc1:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Task 1 - ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve_task1.png")
    plt.close()

    # 绘制任务2的ROC曲线
    plt.figure(figsize=(8, 6))
    fpr2, tpr2, _ = roc_curve(labels_task2, probs_task2)
    roc_auc2 = auc(fpr2, tpr2)
    plt.plot(fpr2, tpr2, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc2:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Task 2 - ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve_task2.png")
    plt.close()

if __name__ == "__main__":
    evaluate_model("model/multitask/twitter_model", "data/processed/twitter_test_mtl.csv")
