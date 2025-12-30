# sarcasm_detection.py
import torch
import torch.nn as nn
import sqlite3
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    PreTrainedModel,
    PretrainedConfig
)
from torch.utils.data import Dataset
import pandas as pd
import extract
import os

class Config:
    text_model = "answerdotai/ModernBERT-base"
    max_length = 128
    max_entities = 5
    entity_length = 32
    db_path = "data/nouns/reddit_train.db"
    batch_size = 8
    lr = 5e-5
    epochs = 6

# 新增配置类
class SarcasmConfig(PretrainedConfig):
    model_type = "sarcasm"
    
    def __init__(
        self,
        text_model=Config.text_model,
        **kwargs
    ):
        self.text_model = text_model
        super().__init__(**kwargs)

# 修改模型继承
class SarcasmModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.text_encoder = AutoModel.from_pretrained(config.text_model)
        
        self.attn = nn.MultiheadAttention(768, 8, batch_first=True, dropout=0.1)  # 添加注意力dropout
        self.gate = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)  # 新增Dropout层
        self.classifier = nn.Linear(768, 2)

    def forward(self, text_input_ids, text_attention_mask, 
                entity_input_ids, entity_attention_mask):
        # ========== 文本编码 ==========
        text_out = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        ).last_hidden_state
        
        # ========== 实体编码 ==========
        batch_size = entity_input_ids.size(0)
        entity_inputs = {
            'input_ids': entity_input_ids.view(-1, Config.entity_length),
            'attention_mask': entity_attention_mask.view(-1, Config.entity_length)
        }
        
        # 移除torch.no_grad()允许梯度传播
        entity_out = self.text_encoder(**entity_inputs).last_hidden_state[:, 0]
        entity_out = entity_out.view(batch_size, Config.max_entities, 768)
        
        # 动态注意力掩码处理
        valid_entities = (entity_attention_mask.sum(dim=-1) > 0).float()  # [B, N]
        attn_mask = valid_entities.unsqueeze(1)  # [B, 1, N]
        
        attn_out, _ = self.attn(
            text_out, 
            entity_out, 
            entity_out,
            key_padding_mask=~valid_entities.bool()  # 精确处理无效实体
        )
        
        gate = torch.sigmoid(self.gate(text_out))
        fused = gate * text_out + (1 - gate) * attn_out
        fused = self.dropout(fused[:, 0])  # 添加Dropout
        return self.classifier(fused)


class SarcasmDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.text_tok = AutoTokenizer.from_pretrained(Config.text_model)
        self.conn = sqlite3.connect(Config.db_path)
        self._validate_data()

    def _validate_data(self):
        assert 'label' in self.df.columns, f"Missing label column. Available: {self.df.columns}"
        print(f"Data validation passed. Total samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = int(self.df.iloc[idx]['label'])
        
        # 文本编码
        text_enc = self.text_tok(
            text, 
            padding='max_length',
            max_length=Config.max_length,
            return_tensors='pt',
            truncation=True
        )
        
        # 实体处理
        entities = extract.extract_special_nouns(text)[:Config.max_entities]
        explanations = []
        # print(entities)
        for e in entities:
            cur = self.conn.cursor()
            cur.execute("SELECT explanation FROM terms WHERE term=?", (e.lower(),))
            row = cur.fetchone()
            explanations.append(row[0] if row else "[UNK]")
        
        # 填充空实体
        explanations += [''] * (Config.max_entities - len(entities))
        
        entity_enc = self.text_tok(
            explanations,
            padding='max_length',
            max_length=Config.entity_length,
            return_tensors='pt',
            truncation=True
        )
        
        return {
            'text_input_ids': text_enc['input_ids'].squeeze(0),
            'text_attention_mask': text_enc['attention_mask'].squeeze(0),
            'entity_input_ids': entity_enc['input_ids'].squeeze(0),
            'entity_attention_mask': entity_enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class EntityCollator:
    def __call__(self, batch):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        return {
            'text_input_ids': torch.stack([x['text_input_ids'] for x in batch]).to(device),
            'text_attention_mask': torch.stack([x['text_attention_mask'] for x in batch]).to(device),
            'entity_input_ids': torch.stack([x['entity_input_ids'] for x in batch]).to(device),
            'entity_attention_mask': torch.stack([x['entity_attention_mask'] for x in batch]).to(device),
            'labels': torch.stack([x['labels'] for x in batch]).to(device)
        }

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # 初始化配置
        config = SarcasmConfig()
        
        # 加载数据
        df = pd.read_csv("data/processed/reddit_train.csv")
        print("Data columns:", df.columns.tolist())
        dataset = SarcasmDataset(df)
        
        # 初始化模型
        model = SarcasmModel(config).to(device)
        
        # 配置训练参数
        args = TrainingArguments(
            output_dir="model/entity/reddit_model",
            bf16=True,  # 启用混合精度训练
            gradient_accumulation_steps=2,  # 内存不足时使用梯度累积
            per_device_train_batch_size=Config.batch_size,
            learning_rate=Config.lr,
            num_train_epochs=Config.epochs,
            logging_dir='./logs',
            logging_steps=50,
            save_strategy="epoch",  # 修改保存策略
            save_total_limit=2,
            remove_unused_columns=False,
            load_best_model_at_end=False,
            save_safetensors=True    # 启用安全格式保存
        )
        
        # 开始训练
        trainer = CustomTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=EntityCollator()
        )
        print("Starting training...")
        trainer.train()
        
        # 训练后保存完整模型
        model.save_pretrained(
            "model/entity/reddit_model",
            safe_serialization=True
        )
        
        # 保存分词器
        tokenizer = AutoTokenizer.from_pretrained(Config.text_model)
        tokenizer.save_pretrained("model/entity/reddit_model")
        
        print("模型保存成功！文件列表：")
        print(os.listdir("model/entity/reddit_model"))
        
    except Exception as e:
        print(f"训练失败: {str(e)}")
        raise