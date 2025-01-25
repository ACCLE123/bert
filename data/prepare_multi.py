import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# 加载情感分析模型，设置最大长度
sentiment_pipeline = pipeline("sentiment-analysis", max_length=512, truncation=True)

# 定义情感标签生成函数
def get_sentiment_label(text):
    try:
        result = sentiment_pipeline(text)[0]
        return 1 if result["label"] == "POSITIVE" else 0
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return 0  # 返回默认值

# 处理单个数据集
def process_dataset(input_path, output_path):
    # 加载数据
    df = pd.read_csv(input_path)
    total_rows = len(df)
    
    print(f"Total rows to process: {total_rows}")
    
    # 使用tqdm包装情感标签生成过程
    tqdm.pandas(desc="Generating sentiment labels")
    df["sentiment_label"] = df["text"].progress_apply(get_sentiment_label)
    
    # 保存处理后的数据
    df.to_csv(output_path, index=False)

# 处理所有数据集
datasets = {
    "news": ("processed/news_test.csv", "processed/news_test_mtl.csv"),
    "reddit": ("processed/reddit_test.csv", "processed/reddit_test_mtl.csv"),
    "twitter": ("processed/twitter_test.csv", "processed/twitter_test_mtl.csv"),
}

# 显示总体进度
for name, (input_path, output_path) in tqdm(datasets.items(), desc="Processing datasets"):
    print(f"\nProcessing {name} dataset...")
    process_dataset(input_path, output_path)
    print(f"Saved to {output_path}")
