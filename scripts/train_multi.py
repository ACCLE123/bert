import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# 检查是否支持 MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(df, tokenizer, max_length=128):
    texts = df["text"].tolist()
    labels_task1 = df["label"].tolist()
    labels_task2 = df["sentiment_label"].tolist()

    # 将两个标签组合成一个二维数组
    combined_labels = [[l1, l2] for l1, l2 in zip(labels_task1, labels_task2)]

    # 创建初始编码
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # 转换标签为张量
    labels_tensor = torch.tensor(combined_labels)

    # 创建数据集
    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"].numpy(),
        "attention_mask": encodings["attention_mask"].numpy(),
        "labels": labels_tensor.numpy()
    })

    # 设置格式
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return dataset


# 可能还需要修改模型的 forward 方法来适应新的输入格式
class MultiTaskBertModel(nn.Module):
    def __init__(self, bert_model_name, num_labels_task1, num_labels_task2):
        super(MultiTaskBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier_task1 = nn.Linear(self.bert.config.hidden_size, num_labels_task1)
        self.classifier_task2 = nn.Linear(self.bert.config.hidden_size, num_labels_task2)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits_task1 = self.classifier_task1(pooled_output)
        logits_task2 = self.classifier_task2(pooled_output)
        return logits_task1, logits_task2

class MultiTaskTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        """
        重写获取训练数据加载器的方法
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        dataloader_params = {
            "batch_size": self.args.train_batch_size,
            "shuffle": True,
            "collate_fn": data_collator,
        }
        
        return DataLoader(train_dataset, **dataloader_params)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 打印调试信息
        # print("Available keys in inputs:", inputs.keys())
        # print("Labels shape:", inputs["labels"].shape if "labels" in inputs else "No labels")
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits_task1, logits_task2 = outputs

        # 确保标签形状正确
        if len(labels.shape) == 1:
            labels = labels.view(-1, 2)

        labels_task1 = labels[:, 0]
        labels_task2 = labels[:, 1]

        # Compute losses
        loss_fct = nn.CrossEntropyLoss()
        loss_task1 = loss_fct(logits_task1.view(-1, 2), labels_task1.view(-1))
        loss_task2 = loss_fct(logits_task2.view(-1, 2), labels_task2.view(-1))

        # Total loss
        loss = loss_task1 + loss_task2

        return (loss, outputs) if return_outputs else loss

def train_model(train_data, test_data, model_save_path):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = MultiTaskBertModel("bert-base-uncased", num_labels_task1=2, num_labels_task2=2).to(device)

    # 准备数据集
    train_dataset = preprocess_data(train_data, tokenizer)
    test_dataset = preprocess_data(test_data, tokenizer)

    # 定义数据整理函数
    def collate_fn(examples):
        batch = {}
        for key in examples[0].keys():
            batch[key] = torch.stack([example[key] for example in examples])
        return batch

    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir="./experiments/logs",
        logging_steps=10,
        eval_strategy="epoch",  # 使用新的参数名
        save_strategy="epoch",
        save_total_limit=2
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn  # 使用自定义的数据整理函数
    )

    # 训练模型
    trainer.train()
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)



if __name__ == "__main__":
    # 示例：训练新闻标题数据
    train_data, test_data = load_data("data/processed/twitter_train_mtl.csv", "data/processed/twitter_test_mtl.csv")
    train_model(train_data, test_data, "model/multitask/twitter_model")