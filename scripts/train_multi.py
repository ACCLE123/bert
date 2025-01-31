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
    labels_sarcasm = df["label"].tolist()
    labels_hostile = df["hostile_label"].tolist()
    labels_contempt = df["contempt_label"].tolist()
    labels_humor = df["humor_label"].tolist()

    # 将四个标签组合成一个二维数组
    combined_labels = [[s, h, c, hu] for s, h, c, hu in zip(labels_sarcasm, 
                                                           labels_hostile, 
                                                           labels_contempt, 
                                                           labels_humor)]

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

class MultiTaskBertModel(nn.Module):
    def __init__(self, bert_model_name):
        super(MultiTaskBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        
        # 四个二分类任务的分类器
        self.classifier_sarcasm = nn.Linear(hidden_size, 2)
        self.classifier_hostile = nn.Linear(hidden_size, 2)
        self.classifier_contempt = nn.Linear(hidden_size, 2)
        self.classifier_humor = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # 四个任务的输出
        logits_sarcasm = self.classifier_sarcasm(pooled_output)
        logits_hostile = self.classifier_hostile(pooled_output)
        logits_contempt = self.classifier_contempt(pooled_output)
        logits_humor = self.classifier_humor(pooled_output)
        
        return logits_sarcasm, logits_hostile, logits_contempt, logits_humor

class MultiTaskTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
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
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits_sarcasm, logits_hostile, logits_contempt, logits_humor = outputs

        # 确保标签形状正确
        if len(labels.shape) == 1:
            labels = labels.view(-1, 4)

        # 分离不同任务的标签
        labels_sarcasm = labels[:, 0]
        labels_hostile = labels[:, 1]
        labels_contempt = labels[:, 2]
        labels_humor = labels[:, 3]

        # 计算每个任务的损失
        loss_fct = nn.CrossEntropyLoss()
        loss_sarcasm = loss_fct(logits_sarcasm.view(-1, 2), labels_sarcasm.view(-1))
        loss_hostile = loss_fct(logits_hostile.view(-1, 2), labels_hostile.view(-1))
        loss_contempt = loss_fct(logits_contempt.view(-1, 2), labels_contempt.view(-1))
        loss_humor = loss_fct(logits_humor.view(-1, 2), labels_humor.view(-1))

        total_loss = 2.0 * loss_sarcasm + 0.3 * (loss_hostile + loss_contempt + loss_humor)

        if return_outputs:
            return total_loss, outputs
        return total_loss

def train_model(train_data, test_data, model_save_path):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = MultiTaskBertModel("bert-base-uncased").to(device)

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
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn
    )

    # 训练模型
    trainer.train()
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

if __name__ == "__main__":
    train_data, test_data = load_data(
        "data/processed/news_train_mtl.csv", 
        "data/processed/news_test_mtl.csv"
    )
    train_model(train_data, test_data, "model/multitask/news_model")
