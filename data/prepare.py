import json
import pandas as pd
from sklearn.model_selection import train_test_split

def load_news_data(file_path):
    """
    加载新闻标题数据集
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append({
                "text": item["headline"],
                "label": item["is_sarcastic"]
            })
    return pd.DataFrame(data)

def load_reddit_data(train_file_path, test_file_path):
    """
    加载Reddit评论数据集（训练集和测试集）
    """
    def load_single_file(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                context = " ".join(item["context"])
                response = item["response"]
                text = f"{context} {response}"  # 将上下文和回复拼接
                label = 1 if item["label"] == "SARCASM" else 0
                data.append({
                    "text": text,
                    "label": label
                })
        return pd.DataFrame(data)

    train_df = load_single_file(train_file_path)
    test_df = load_single_file(test_file_path)
    return train_df, test_df

def load_twitter_data(train_file_path, test_file_path):
    """
    加载Twitter推文数据集（训练集和测试集）
    """
    def load_single_file(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                context = " ".join(item["context"])
                response = item["response"]
                text = f"{context} {response}"  # 将上下文和回复拼接
                label = 1 if item["label"] == "SARCASM" else 0
                data.append({
                    "text": text,
                    "label": label
                })
        return pd.DataFrame(data)

    train_df = load_single_file(train_file_path)
    test_df = load_single_file(test_file_path)
    return train_df, test_df

def save_data(df, file_path):
    """
    保存处理后的数据
    """
    df.to_csv(file_path, index=False, encoding='utf-8')

def main():
    # 加载新闻标题数据
    news_df = load_news_data("news/Sarcasm_Headlines_Dataset.json")
    news_train, news_test = train_test_split(news_df, test_size=0.2, random_state=42)
    save_data(news_train, "news_train.csv")
    save_data(news_test, "news_test.csv")
    print(f"新闻标题数据集：训练集大小={len(news_train)}，测试集大小={len(news_test)}")

    # 加载Reddit数据
    reddit_train, reddit_test = load_reddit_data(
        "reddit/sarcasm_detection_shared_task_reddit_training.jsonl",
        "reddit/sarcasm_detection_shared_task_reddit_testing.jsonl"
    )
    save_data(reddit_train, "reddit_train.csv")
    save_data(reddit_test, "reddit_test.csv")
    print(f"Reddit数据集：训练集大小={len(reddit_train)}，测试集大小={len(reddit_test)}")

    # 加载Twitter数据
    twitter_train, twitter_test = load_twitter_data(
        "twitter/sarcasm_detection_shared_task_twitter_training.jsonl",
        "twitter/sarcasm_detection_shared_task_twitter_testing.jsonl"
    )
    save_data(twitter_train, "twitter_train.csv")
    save_data(twitter_test, "twitter_test.csv")
    print(f"Twitter数据集：训练集大小={len(twitter_train)}，测试集大小={len(twitter_test)}")

if __name__ == "__main__":
    main()