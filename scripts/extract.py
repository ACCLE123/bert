import pandas as pd
import spacy
import json
import re
import torch
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# 检查设备兼容性（兼容MPS/Mac）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 初始化spacy（禁用不需要的组件）
nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])

# 初始化NER模型（使用自动类型加载）
try:
    ner = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        tokenizer=AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english"),
        device=device if device.type != 'cpu' else -1,  # 兼容MPS
        aggregation_strategy="first"
    )
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    ner = None

DATASETS = [
    "reddit_test",
    "reddit_train",
    "twitter_test",
    "twitter_train"
]

def clean_text(text):
    """优化后的文本清洗函数"""
    # 保留可能包含讽刺意义的符号（如问号、感叹号）
    text = re.sub(r'@\w+|#\w+', '', text)  # 移除社交媒体标签
    text = re.sub(r'http\S+', '', text)     # 移除URL
    text = re.sub(r'[^\w\s\-\'?.!]', '', text)  # 保留基本标点
    text = re.sub(r'\s+', ' ', text)       # 合并多余空格
    return text.strip()

def extract_special_nouns(text):
    """增强版实体提取"""
    if not ner:
        return []
    
    try:
        entities = ner(text)
        valid_phrases = []
        current_phrase = []
        
        for entity in entities:
            # 处理子词合并
            word = entity['word'].replace('##', '')
            
            # 实体连续性检测
            if current_phrase and 'start' in entity and 'end' in entity:
                if entity['start'] != current_phrase[-1].get('end', entity['start']):
                    valid_phrases.append(current_phrase)
                    current_phrase = []
                
            current_phrase.append({
                'word': word,
                'entity': entity['entity_group'],
                'score': entity['score'],
                'start': entity.get('start', 0),  # 添加默认值
                'end': entity.get('end', 0)       # 添加默认值
            })
        
        if current_phrase:
            valid_phrases.append(current_phrase)
        
        # 过滤和合并
        results = set()
        for phrase in valid_phrases:
            avg_score = sum(e['score'] for e in phrase) / len(phrase)
            if avg_score > 0.85 and len(phrase) <= 3:  # 限制短语长度
                full_phrase = ' '.join([e['word'] for e in phrase])
                if 3 <= len(full_phrase) <= 40:
                    results.add(full_phrase)
        return list(results)
    
    except Exception as e:
        print(f"实体提取失败: {str(e)}")
        return []
    

def process_dataset(dataset_name):
    """数据处理流程"""
    input_path = Path(f"data/processed/{dataset_name}.csv")
    output_path = Path(f"data/nouns/{dataset_name}_nouns.json")
    
    # 文件检查
    if not input_path.exists():
        print(f"⚠️ 文件不存在: {input_path}")
        return
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 读取数据
        df = pd.read_csv(input_path, usecols=['text'], dtype={'text': str})
        print(f"处理数据集: {dataset_name} ({len(df)} 条)")
        
        # 并行处理
        texts = df['text'].dropna().apply(clean_text).tolist()
        all_nouns = set()
        
        # 批量处理提升性能
        batch_size = 64 if device.type == 'cuda' else 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                if text:
                    all_nouns.update(extract_special_nouns(text))
        
        # 保存结果
        with open(output_path, 'w') as f:
            json.dump(sorted(all_nouns), f, indent=2)
            
        print(f"✅ 保存 {len(all_nouns)} 个名词到 {output_path}")
    
    except Exception as e:
        print(f"处理失败: {str(e)}")

if __name__ == "__main__":
    if not ner:
        print("❌ NER模型未加载，请检查模型路径或网络连接")
    else:
        print(f"运行设备: {device}")
        for dataset in DATASETS:
            process_dataset(dataset)
        print("\n处理完成！")