# inference.py
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_entity import SarcasmModel, SarcasmDataset, EntityCollator
from transformers import AutoTokenizer
import sqlite3
import os
from sklearn.metrics import accuracy_score, f1_score  # 新增导入
from train_entity import Config
import extract

class InferenceConfig:
    db_path = "data/nouns/reddit_test.db"

def predict(model, dataloader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="推理进度"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = model(**inputs)
            probs = torch.softmax(outputs, dim=1)
            predictions.extend(probs.cpu().numpy())
    return predictions

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 数据加载
        test_df = pd.read_csv("data/processed/reddit_test.csv")
        os.makedirs("results", exist_ok=True)

        # 数据集初始化
        class TestDataset(SarcasmDataset):
            def __init__(self, df):
                self.df = df.reset_index(drop=True)
                self.text_tok = AutoTokenizer.from_pretrained(Config.text_model)
                self.conn = sqlite3.connect(InferenceConfig.db_path)  # 直接连接测试库
            def __getitem__(self, idx):
                text = self.df.iloc[idx]['text']
                label = int(self.df.iloc[idx].get('label', -1))
                
                # ==== 实体提取 ====
                entities = extract.extract_special_nouns(text)[:Config.max_entities]
                explanations = []
                for e in entities:
                    cur = self.conn.cursor()
                    cur.execute("SELECT explanation FROM terms WHERE term=?", (e.lower(),))
                    row = cur.fetchone()
                    explanations.append(row[0] if row else "[UNK]")
                # 填充空实体
                explanations += [''] * (Config.max_entities - len(entities))
                
                # ==== 实体编码 ====
                entity_enc = self.text_tok(
                    explanations,
                    padding='max_length',
                    max_length=Config.entity_length,
                    return_tensors='pt',
                    truncation=True
                )
                
                # ==== 文本编码 ====
                text_enc = self.text_tok(
                    text, 
                    padding='max_length',
                    max_length=Config.max_length,
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
        test_dataset = TestDataset(test_df)
        
        # 数据加载器
        collator = EntityCollator()
        collator.device = device
        dataloader = DataLoader(
            test_dataset,
            batch_size=32,
            collate_fn=collator,
            shuffle=False
        )

        # 模型加载
        model = SarcasmModel.from_pretrained("model/entity/reddit_model")
        model = model.to(device)
        model.eval()

        # 执行推理
        results = predict(model, dataloader, device)

        # 保存结果
        test_df['sarcasm_prob'] = [float(p[1]) for p in results]
        test_df['prediction'] = [int(p.argmax()) for p in results]
        test_df.to_csv("results/predictions.csv", index=False)

        # 评估指标计算
        if 'label' in test_df.columns:
            true_labels = test_df['label'].values.astype(int)
            pred_labels = test_df['prediction'].values.astype(int)
            
            accuracy = accuracy_score(true_labels, pred_labels)
            f1 = f1_score(true_labels, pred_labels, average='binary')
            print(accuracy)
            print(f1)
        else:
            print("警告: 测试数据未包含标签列 'label'")

        print("推理完成！结果保存在 results/ 目录")

    except Exception as e:
        print(f"推理失败: {str(e)}")
        raise