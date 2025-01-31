from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import BertTokenizer
from train_multi import MultiTaskBertModel
import pandas as pd
import numpy as np
from safetensors.torch import load_file
from sklearn.metrics import classification_report

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def evaluate_model(model_path, test_data_path):
    # 加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = MultiTaskBertModel("bert-base-uncased")
    state_dict = load_file(f"{model_path}/model.safetensors")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 加载测试数据
    test_df = pd.read_csv(test_data_path)
    texts = test_df["text"].tolist()
    labels_sarcasm = test_df["label"].tolist()
    labels_hostile = test_df["hostile_label"].tolist()
    labels_contempt = test_df["contempt_label"].tolist()
    labels_humor = test_df["humor_label"].tolist()

    # 数据预处理
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    inputs = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # 模型预测
    with torch.no_grad():
        logits_sarcasm, logits_hostile, logits_contempt, logits_humor = model(inputs, attention_mask=attention_mask)
        
        # 获取预测结果
        predictions_sarcasm = torch.argmax(logits_sarcasm, dim=1).cpu().numpy()
        predictions_hostile = torch.argmax(logits_hostile, dim=1).cpu().numpy()
        predictions_contempt = torch.argmax(logits_contempt, dim=1).cpu().numpy()
        predictions_humor = torch.argmax(logits_humor, dim=1).cpu().numpy()
        
        # 获取概率
        probs_sarcasm = torch.softmax(logits_sarcasm, dim=1).cpu().numpy()[:, 1]
        probs_hostile = torch.softmax(logits_hostile, dim=1).cpu().numpy()[:, 1]
        probs_contempt = torch.softmax(logits_contempt, dim=1).cpu().numpy()[:, 1]
        probs_humor = torch.softmax(logits_humor, dim=1).cpu().numpy()[:, 1]

    # 评估每个任务
    tasks = {
        "Sarcasm Detection": (labels_sarcasm, predictions_sarcasm, probs_sarcasm),
        "Hostile Detection": (labels_hostile, predictions_hostile, probs_hostile),
        "Contempt Detection": (labels_contempt, predictions_contempt, probs_contempt),
        "Humor Detection": (labels_humor, predictions_humor, probs_humor)
    }

    for task_name, (labels, preds, probs) in tasks.items():
        print(f"\n{task_name} Results:")
        
        # 检查是否存在预测类别
        unique_preds = np.unique(preds)
        if len(unique_preds) == 1:
            print(f"Warning: Model only predicts class {unique_preds[0]} for {task_name}")
            
        accuracy = accuracy_score(labels, preds)
        # 使用zero_division=0来处理f1_score的警告
        f1 = f1_score(labels, preds, zero_division=0)
        print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

        # 绘制混淆矩阵
        plt.figure(figsize=(6, 6))
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Class 0", "Class 1"],
                    yticklabels=["Class 0", "Class 1"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{task_name} - Confusion Matrix")
        plt.savefig(f"confusion_matrix_{task_name.lower().replace(' ', '_')}.png")
        plt.close()

        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{task_name} - ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig(f"roc_curve_{task_name.lower().replace(' ', '_')}.png")
        plt.close()

        # 打印分类报告，添加zero_division=0参数
        print(f"\nDetailed Classification Report for {task_name}:")
        print(classification_report(labels, preds, 
                                 target_names=["Class 0", "Class 1"],
                                 zero_division=0))

    # 打印类别分布信息
    print("\nClass Distribution in Test Set:")
    for task_name, (labels, _, _) in tasks.items():
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"{task_name}:")
        print(f"Class 0: {dist.get(0, 0)}, Class 1: {dist.get(1, 0)}")
        print(f"Ratio (Class 1/Total): {dist.get(1, 0)/len(labels):.3f}")

if __name__ == "__main__":
    evaluate_model("model/multitask/news_model", "data/processed/news_test_mtl.csv")
