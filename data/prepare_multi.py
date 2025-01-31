import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# 加载情感分析模型
emotion_pipeline = pipeline("text-classification", 
                          model="SamLowe/roberta-base-go_emotions", 
                          max_length=512, 
                          truncation=True)

def get_emotion_labels(text):
    try:
        # 获取情感分析结果
        emotion_results = emotion_pipeline(text)
        result = emotion_results[0]  # 获取预测结果
        label = result['label']
        score = result['score']
        
        # 定义情感类别映射
        hostile_emotions = {'anger', 'annoyance', 'disgust'}
        contempt_emotions = {'disapproval', 'disappointment'}
        humor_emotions = {'amusement', 'joy', 'excitement'}
        
        # 设置阈值
        threshold = 0.5
        
        return {
            'hostile': 1 if (label in hostile_emotions and score > threshold) else 0,
            'contempt': 1 if (label in contempt_emotions and score > threshold) else 0,
            'humor': 1 if (label in humor_emotions and score > threshold) else 0
        }
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return {
            'hostile': 0,
            'contempt': 0,
            'humor': 0
        }

def process_dataset(input_path, output_path):
    # 加载数据
    df = pd.read_csv(input_path)
    total_rows = len(df)
    
    print(f"Processing {total_rows} rows...")
    
    # 使用tqdm显示进度
    tqdm.pandas(desc="Generating emotion labels")
    
    # 应用情感分析并展开结果
    emotion_results = df["text"].progress_apply(get_emotion_labels)
    
    # 将情感分析结果添加为新列
    df['hostile_label'] = emotion_results.apply(lambda x: x['hostile'])
    df['contempt_label'] = emotion_results.apply(lambda x: x['contempt'])
    df['humor_label'] = emotion_results.apply(lambda x: x['humor'])
    
    # 保存处理后的数据
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

# 处理所有数据集
datasets = {
    "news_train": ("processed/news_train.csv", "processed/news_train_mtl.csv"),
    "news_test": ("processed/news_test.csv", "processed/news_test_mtl.csv"),
    "reddit_train": ("processed/reddit_train.csv", "processed/reddit_train_mtl.csv"),
    "reddit_test": ("processed/reddit_test.csv", "processed/reddit_test_mtl.csv"),
    "twitter_train": ("processed/twitter_train.csv", "processed/twitter_train_mtl.csv"),
    "twitter_test": ("processed/twitter_test.csv", "processed/twitter_test_mtl.csv"),
}

# 显示总体进度
for name, (input_path, output_path) in tqdm(datasets.items(), desc="Processing datasets"):
    print(f"\nProcessing {name} dataset...")
    process_dataset(input_path, output_path)
