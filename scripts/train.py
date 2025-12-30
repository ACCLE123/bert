import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch

# 检查是否支持 MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

# 数据预处理
def preprocess_data(df, tokenizer, max_length=128):
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # 使用 datasets 库创建 Dataset
    dataset = Dataset.from_dict({
        "text": texts,
        "label": labels
    })

    # Tokenize 数据
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# 训练模型
def train_model(train_data, test_data, model_save_path):
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=2).to(device)
    
    train_encodings = preprocess_data(train_data, tokenizer)
    test_encodings = preprocess_data(test_data, tokenizer)

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir="./experiments/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
    )

    # 定义Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encodings,
        eval_dataset=test_encodings,
    )

    # 训练并保存模型
    trainer.train()
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)  # 保存分词器


if __name__ == "__main__":
    train_data, test_data = load_data("data/processed/reddit_train.csv", "data/processed/reddit_test.csv")
    train_model(train_data, test_data, "model/base/reddit_model")