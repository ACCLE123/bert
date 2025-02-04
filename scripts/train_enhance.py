import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import List, Dict, Union, Any 
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from nltk.corpus import sentiwordnet as swn
import emoji
import re
from typing import List, Dict
import json
import nltk
from transformers import AutoModel

def ensure_nltk_resources():
    resources = ['wordnet', 'sentiwordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource)
            print(f"Downloaded {resource} successfully!")

ensure_nltk_resources()

# 检查设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class KnowledgeBase:
    def __init__(self):
        self.sentiment_lexicon = self._load_sentiment_lexicon()
        self.slang_dict = self._load_slang_dictionary()
        self.emoji_dict = self._load_emoji_meanings()
        
    def _load_sentiment_lexicon(self):
        sentiment_dict = {}
        for senti_synset in list(swn.all_senti_synsets()):
            word = senti_synset.synset.lemmas()[0].name()
            pos_score = senti_synset.pos_score()
            neg_score = senti_synset.neg_score()
            sentiment_dict[word] = {
                'positive': pos_score,
                'negative': neg_score
            }
        return sentiment_dict
    
    def _load_slang_dictionary(self):
        return {
            "lol": "laughing out loud",
            "idk": "i don't know",
            "tbh": "to be honest",
            "imo": "in my opinion",
            "goat": "greatest of all time",
            "sick": "excellent",
            "lit": "exciting",
            "fam": "family",
            "ngl": "not gonna lie",
            "fr": "for real",
            "rn": "right now",
            "idc": "i don't care",
            "gg": "good game",
            "dm": "direct message",
            "fomo": "fear of missing out"
        }
    
    def _load_emoji_meanings(self):
        return {
            "🔥": "fire|exciting|excellent",
            "😊": "happy|pleased|positive",
            "😢": "sad|unhappy|negative",
            "👍": "approve|good|positive",
            "👎": "disapprove|bad|negative",
            "❤️": "love|adore|positive",
            "😂": "laughing|amused|positive",
            "😡": "angry|mad|negative",
            "🙄": "annoyed|skeptical|negative",
            "💪": "strong|confident|positive"
        }

class EnhancedPreprocessor:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        
    def preprocess_text(self, text: str) -> Dict:
        text = text.lower().strip()
        
        sentiment_info = self._extract_sentiment_info(text)
        slang_info = self._handle_slang(text)
        emoji_info = self._handle_emojis(text)
        
        # 处理文本
        text = self._replace_slang(text)
        text = self._clean_emojis(text)
        
        return {
            'processed_text': text,
            'sentiment_features': sentiment_info,
            'slang_features': slang_info,
            'emoji_features': emoji_info
        }
    
    def _extract_sentiment_info(self, text: str) -> Dict:
        words = text.split()
        sentiment_scores = {
            'positive': 0.0,
            'negative': 0.0
        }
        
        for word in words:
            if word in self.knowledge_base.sentiment_lexicon:
                scores = self.knowledge_base.sentiment_lexicon[word]
                sentiment_scores['positive'] += scores['positive']
                sentiment_scores['negative'] += scores['negative']
                
        return sentiment_scores
    
    def _handle_slang(self, text: str) -> Dict:
        words = text.split()
        slang_replacements = {}
        
        for word in words:
            if word in self.knowledge_base.slang_dict:
                slang_replacements[word] = self.knowledge_base.slang_dict[word]
                
        return slang_replacements
    
    def _handle_emojis(self, text: str) -> Dict:
        emoji_features = {}
        for char in text:
            if char in self.knowledge_base.emoji_dict:
                emoji_features[char] = self.knowledge_base.emoji_dict[char]
        return emoji_features
    
    def _replace_slang(self, text: str) -> str:
        words = text.split()
        for i, word in enumerate(words):
            if word in self.knowledge_base.slang_dict:
                words[i] = self.knowledge_base.slang_dict[word]
        return ' '.join(words)
    
    def _clean_emojis(self, text: str) -> str:
        for emoji_char in self.knowledge_base.emoji_dict:
            if emoji_char in text:
                text = text.replace(emoji_char, f" {self.knowledge_base.emoji_dict[emoji_char]} ")
        return text

class KnowledgeEnhancedBERT(nn.Module):
    def __init__(self, model_name="answerdotai/ModernBERT-base", knowledge_dim=64):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.knowledge_projection = nn.Linear(3 * knowledge_dim, knowledge_dim)
        self.final_classifier = nn.Linear(self.bert.config.hidden_size + knowledge_dim, 2)
        
        # 正确初始化权重
        nn.init.xavier_normal_(self.knowledge_projection.weight)
        nn.init.xavier_normal_(self.final_classifier.weight)
        nn.init.zeros_(self.knowledge_projection.bias)
        nn.init.zeros_(self.final_classifier.bias)
        
    def forward(self, input_ids, attention_mask, knowledge_features, labels=None):
        # BERT 输出
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_pooled = bert_output.last_hidden_state[:, 0, :]
        
        # 知识特征处理
        knowledge_vector = self.knowledge_projection(knowledge_features)
        
        # 特征融合
        combined_features = torch.cat([bert_pooled, knowledge_vector], dim=1)
        
        # 最终分类
        logits = self.final_classifier(combined_features)
        
        # 如果提供了标签，计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return {'loss': loss, 'logits': logits}
            
        return logits


def prepare_knowledge_features(preprocessed_data: Dict) -> torch.Tensor:
    # 提取情感分数
    sentiment_features = [
        preprocessed_data['sentiment_features']['positive'],
        preprocessed_data['sentiment_features']['negative']
    ]
    
    # 统计特征
    slang_count = len(preprocessed_data['slang_features'])
    emoji_count = len(preprocessed_data['emoji_features'])
    
    # 创建特征向量
    knowledge_features = [0.0] * 192  # 3 * knowledge_dim (64)
    knowledge_features[0] = float(sentiment_features[0])
    knowledge_features[1] = float(sentiment_features[1])
    knowledge_features[2] = float(slang_count)
    knowledge_features[3] = float(emoji_count)
    
    return torch.tensor(knowledge_features, dtype=torch.float)

class DataCollatorForKnowledgeEnhanced:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        try:
            # 确保所有输入都是列表
            input_ids = [f['input_ids'] if isinstance(f['input_ids'], list) 
                        else f['input_ids'].tolist() for f in features]
            attention_mask = [f['attention_mask'] if isinstance(f['attention_mask'], list)
                            else f['attention_mask'].tolist() for f in features]
            knowledge_features = [f['knowledge_features'] if isinstance(f['knowledge_features'], list)
                                else f['knowledge_features'].tolist() for f in features]
            
            # 转换为张量
            batch = {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'knowledge_features': torch.tensor(knowledge_features, dtype=torch.float),
                'labels': torch.tensor([f['labels'] for f in features], dtype=torch.long)
            }
            return batch
            
        except Exception as e:
            print("Error in DataCollator:")
            print(f"Features: {features}")
            print(f"Error message: {str(e)}")
            raise e

def prepare_dataset(df):
    processed_data = []
    for _, row in df.iterrows():
        try:
            text = str(row['text'])
            label = int(row['label'])
            
            # 知识增强预处理
            enhanced_features = preprocessor.preprocess_text(text)
            
            # tokenization
            encoding = tokenizer(
                enhanced_features['processed_text'],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors=None  # 返回列表
            )
            
            # 准备知识特征
            knowledge_features = prepare_knowledge_features(enhanced_features).tolist()  # 转换为列表
            
            processed_data.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'knowledge_features': knowledge_features,
                'labels': label
            })
            
        except Exception as e:
            print(f"Error processing row: {row}")
            print(f"Error message: {str(e)}")
            continue
    
    return Dataset.from_list(processed_data)


class KnowledgeEnhancedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        try:
            # 将所有输入移到正确的设备上
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 前向传播
            outputs = model(**inputs)
            
            # 如果返回的是字典（包含损失），直接使用
            if isinstance(outputs, dict):
                loss = outputs['loss']
                if num_items_in_batch is not None:
                    loss = loss * (inputs["input_ids"].size(0) / num_items_in_batch)
                return (loss, outputs) if return_outputs else loss
                
            # 如果只返回 logits，需要计算损失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(outputs, inputs['labels'])
            
            if num_items_in_batch is not None:
                loss = loss * (inputs["input_ids"].size(0) / num_items_in_batch)

            return (loss, {'logits': outputs}) if return_outputs else loss
            
        except Exception as e:
            print(f"Error in compute_loss: {str(e)}")
            print("Input shapes:")
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k}: {v.shape}")
            raise e

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        if isinstance(outputs, dict):
            loss = outputs['loss'].mean().detach() if 'loss' in outputs else None
            logits = outputs['logits']
        else:
            loss = None
            logits = outputs
            
        labels = inputs.get('labels')
        
        return loss, logits, labels


def validate_dataset(dataset):
    """验证数据集是否包含所有必需的字段"""
    required_fields = ['input_ids', 'attention_mask', 'knowledge_features', 'labels']
    
    for i, item in enumerate(dataset):
        for field in required_fields:
            if field not in item:
                raise ValueError(f"Missing required field '{field}' in dataset item {i}")
            
        # 验证数据类型
        if not isinstance(item['labels'], (int, torch.Tensor)):
            raise ValueError(f"Invalid label type in item {i}: {type(item['labels'])}")

def train_enhanced_model(train_data, test_data, model_save_path):
    print("Initializing model and tokenizer...")
    
    # 初始化全局变量
    global tokenizer, preprocessor
    
    # 初始化知识库和预处理器
    knowledge_base = KnowledgeBase()
    preprocessor = EnhancedPreprocessor(knowledge_base)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # 创建增强模型
    model = KnowledgeEnhancedBERT("answerdotai/ModernBERT-base").to(device)
    
    print("Preparing datasets...")
    train_dataset = prepare_dataset(train_data)
    test_dataset = prepare_dataset(test_data)
    
    # 验证数据集
    print("Validating datasets...")
    validate_dataset(train_dataset)
    validate_dataset(test_dataset)
    
    # 创建数据整理器
    data_collator = DataCollatorForKnowledgeEnhanced(tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./experiments/logs",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",  # 修改这里
        eval_steps=100,
        save_total_limit=2,
        learning_rate=1e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        gradient_accumulation_steps=2,
        warmup_steps=500,
        fp16=False,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        greater_is_better=False
    )
    
    # 初始化训练器
    trainer = KnowledgeEnhancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )
    
    # 训练模型
    print("Starting training...")
    trainer.train()
    
    # 保存模型
    print(f"Saving model to {model_save_path}")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print("Training completed!")


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

if __name__ == "__main__":
    # 加载数据
    train_data, test_data = load_data("data/processed/twitter_train.csv", "data/processed/twitter_test.csv")
    train_enhanced_model(train_data, test_data, "model/enhanced/twitter_model")
