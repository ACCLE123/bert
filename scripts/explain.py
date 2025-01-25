import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 检查是否支持 MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def explain_model(model_path, text):
    # 加载模型和分词器
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)

    # 定义预测函数
    def predict_proba(texts):
        encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encodings)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        return probs

    # 创建 LIME 解释器
    explainer = LimeTextExplainer(class_names=["非讽刺", "讽刺"])

    # 解释单个样本
    exp = explainer.explain_instance(text, predict_proba, num_features=10)

    # 输出解释结果
    print("Explanation for class '讽刺':")
    for feature, weight in exp.as_list():
        print(f"{feature}: {weight:.4f}")

    # 保存解释结果到 HTML 文件
    exp.save_to_file("explanation.html")
    print("Explanation saved to 'explanation.html'")

    # 绘制特征权重条形图
    features, weights = zip(*exp.as_list())
    plt.figure(figsize=(10, 6))
    plt.barh(features, weights, color='skyblue')
    plt.xlabel("Weight")
    plt.title("Feature Importance for Sarcasm Prediction")
    plt.savefig("feature_importance.png")  # 保存为图片
    plt.show()

if __name__ == "__main__":
    explain_model("model/base/news_model", "This is a sarcastic headline!")