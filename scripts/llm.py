import requests
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# 1. 加载数据
data = pd.read_csv('data/processed/twitter_test.csv')
texts = data['text'].tolist()
labels = data['label'].tolist()

# 2. 定义请求函数
def predict_sarcasm(text):
    url = "http://localhost:11434/api/generate"
    
    # 修改提示词，明确要求只输出0或1
    prompt = """
    Determine if the following text is sarcastic.
    Rules:
    - You must ONLY respond with the number 0 or 1
    - 1 means sarcastic
    - 0 means not sarcastic
    - No other text or explanation is allowed
    
    Text: {text}
    
    Output:""".format(text=text)
    
    payload = {
        "model": "qwen",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            # 严格检查返回值
            if result == "0":
                return 0
            elif result == "1":
                return 1
            else:
                print(f"Invalid response: {result}, defaulting to 0")
                return 0
        else:
            print(f"Request failed with status code: {response.status_code}")
            return 0
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return 0

# 3. 获取预测结果
predictions = []
for text in tqdm(texts, desc="Predicting", unit="text"):
    predictions.append(predict_sarcasm(text))

# 4. 计算评估指标
accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
