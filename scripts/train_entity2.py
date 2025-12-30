# sarcasm_detection.py
import torch
import torch.nn as nn
import sqlite3
import os
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
from phi3_feature_extractor import Phi3FeatureExtractor
from tqdm import tqdm

# 禁用动态图优化
os.environ["TORCHDYNAMO_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True

class Config:
    text_model = "answerdotai/ModernBERT-base"
    max_length = 128
    max_entities = 5
    entity_length = 32
    db_path = "data/nouns/reddit_train.db"
    batch_size = 8
    lr = 5e-5
    epochs = 6
    phi3_feature_dim = 3072

class SarcasmConfig(PretrainedConfig):
    model_type = "sarcasm"
    
    def __init__(
        self,
        text_model=Config.text_model,
        phi3_feature_dim=Config.phi3_feature_dim,
        **kwargs
    ):
        self.text_model = text_model
        self.phi3_feature_dim = phi3_feature_dim
        super().__init__(**kwargs)

class SarcasmModel(PreTrainedModel):
    config_class = SarcasmConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.text_encoder = AutoModel.from_pretrained(config.text_model)
        
        self.phi3_projector = nn.Sequential(
            nn.Linear(config.phi3_feature_dim, 768),
            nn.LayerNorm(768),
            nn.GELU()
        )
        
        self.attn = nn.MultiheadAttention(768, 8, batch_first=True)
        self.gate = nn.Linear(768, 768)
        
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )

    def forward(self, 
               text_input_ids, 
               text_attention_mask, 
               entity_input_ids, 
               entity_attention_mask,
               phi3_features):
        # 确保所有输入在相同设备
        device = text_input_ids.device
        
        # 文本编码
        text_out = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        ).last_hidden_state

        # 实体编码
        batch_size = entity_input_ids.size(0)
        entity_inputs = {
            'input_ids': entity_input_ids.view(-1, Config.entity_length),
            'attention_mask': entity_attention_mask.view(-1, Config.entity_length)
        }
        entity_out = self.text_encoder(**entity_inputs).last_hidden_state[:, 0]
        entity_out = entity_out.view(batch_size, Config.max_entities, 768).to(device)

        # 特征融合
        attn_out, _ = self.attn(text_out, entity_out, entity_out)
        gate = torch.sigmoid(self.gate(text_out))
        fused_base = gate * text_out + (1 - gate) * attn_out

        # Phi3特征处理
        projected_phi3 = self.phi3_projector(phi3_features.to(torch.float32))  # 强制类型转换
        
        combined = torch.cat([fused_base[:, 0], projected_phi3], dim=1)
        return self.classifier(combined)

class SarcasmDataset(Dataset):
    def __init__(self, df, phi3_extractor: Phi3FeatureExtractor, device):
        self.df = df.reset_index(drop=True)
        self.text_tok = AutoTokenizer.from_pretrained(Config.text_model)
        self.conn = sqlite3.connect(Config.db_path)
        self.phi3_extractor = phi3_extractor
        self.device = device
        self._preload_entity_db()
        self._precompute_features()

    def _preload_entity_db(self):
        """预加载实体数据库到内存"""
        self.entity_db = {}
        cur = self.conn.cursor()
        cur.execute("SELECT term, explanation FROM terms")
        for term, explanation in cur.fetchall():
            self.entity_db[term.lower()] = explanation

    def _precompute_features(self):
        """批量预计算Phi-3特征"""
        print("预计算Phi-3特征...")
        texts = self.df['text'].tolist()
        batch_size = 64  # 根据显存调整
        
        self.phi3_features = []
        for i in tqdm(range(0, len(texts), batch_size), 
                        desc="Phi-3特征提取",
                        unit="batch"):
            batch = texts[i:i+batch_size]
            features = self.phi3_extractor.batch_extract(batch)
            self.phi3_features.append(features.to(self.device))
        
        self.phi3_features = torch.cat(self.phi3_features)
        assert self.phi3_features.device == self.device

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
        explanations = [self.entity_db.get(e.lower(), "[UNK]") for e in entities]
        explanations += [''] * (Config.max_entities - len(entities))
        
        entity_enc = self.text_tok(
            explanations,
            padding='max_length',
            max_length=Config.entity_length,
            return_tensors='pt',
            truncation=True
        )
        
        return {
            'text_input_ids': text_enc['input_ids'].squeeze(0).to(self.device),
            'text_attention_mask': text_enc['attention_mask'].squeeze(0).to(self.device),
            'entity_input_ids': entity_enc['input_ids'].squeeze(0).to(self.device),
            'entity_attention_mask': entity_enc['attention_mask'].squeeze(0).to(self.device),
            'phi3_features': self.phi3_features[idx],
            'labels': torch.tensor(label, dtype=torch.long).to(self.device)
        }

class EntityCollator:
    def __call__(self, batch):
        return {
            'text_input_ids': torch.stack([x['text_input_ids'] for x in batch]),
            'text_attention_mask': torch.stack([x['text_attention_mask'] for x in batch]),
            'entity_input_ids': torch.stack([x['entity_input_ids'] for x in batch]),
            'entity_attention_mask': torch.stack([x['entity_attention_mask'] for x in batch]),
            'phi3_features': torch.stack([x['phi3_features'] for x in batch]),
            'labels': torch.stack([x['labels'] for x in batch])
        }

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        phi3_features = inputs.pop('phi3_features')
        outputs = model(phi3_features=phi3_features, **inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    device = torch.device("mps")
    print(f"使用设备: {device}")
    
    try:
        phi3_extractor = Phi3FeatureExtractor()
        df = pd.read_csv("data/processed/reddit_train.csv")
        
        # 初始化数据集（传递device参数）
        dataset = SarcasmDataset(df, phi3_extractor, device)
        
        # 初始化模型
        config = SarcasmConfig()
        model = SarcasmModel(config).to(device)
        
        # 配置训练参数
        args = TrainingArguments(
            output_dir="model/entity2/reddit_model",
            per_device_train_batch_size=Config.batch_size,
            learning_rate=Config.lr,
            num_train_epochs=Config.epochs,
            logging_dir='./logs',
            logging_steps=50,
            save_strategy="epoch",
            save_total_limit=2,
            remove_unused_columns=False,
            fp16=True,  # 使用fp16混合精度
            gradient_accumulation_steps=2,
            dataloader_pin_memory=True,  # 启用内存锁页
            optim="adamw_torch_fused",  # 使用融合优化器
        )

        # 开始训练
        trainer = CustomTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=EntityCollator()
        )
        print("开始训练...")
        trainer.train()
        
        # 保存模型
        model.save_pretrained("model/entity2/reddit_model")
        tokenizer = AutoTokenizer.from_pretrained(Config.text_model)
        tokenizer.save_pretrained("model/entity2/reddit_model")
        
    except Exception as e:
        print(f"训练失败: {str(e)}")
        raise