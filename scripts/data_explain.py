import json
import sqlite3
from pathlib import Path
from tqdm import tqdm
import explorer
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

class NounDatabaseBuilderOptimized:
    def __init__(self, max_workers=8, batch_size=50):
        self.explorer = explorer.WikidataExplorer()
        self.base_path = Path("./data/nouns")
        self.max_workers = max_workers  # 并行线程数
        self.batch_size = batch_size    # 批量插入大小
        
        self.file_map = [
            ("twitter_test_nouns.json", "twitter_test.db"),
            ("twitter_train_nouns.json", "twitter_train.db")
        ]

    def create_table(self, conn):
        conn.execute('''CREATE TABLE IF NOT EXISTS terms (
                        term TEXT PRIMARY KEY,
                        explanation TEXT NOT NULL)''')

    @lru_cache(maxsize=5000)  # 缓存已查询术语
    def get_explanation(self, term):
        results = self.explorer.explore_term(term)
        return self.explorer.format_relation_graph(results, term)

    def process_batch(self, conn, terms):
        # 批量插入数据
        terms = [(t.strip(), self.get_explanation(t)) for t in terms]
        conn.executemany('''
            INSERT OR IGNORE INTO terms (term, explanation)
            VALUES (?, ?)
        ''', terms)
        conn.commit()

    def process_file(self, json_file, db_file):
        json_path = self.base_path / json_file
        db_path = self.base_path / db_file
        
        with sqlite3.connect(db_path) as conn:
            self.create_table(conn)
            
            with open(json_path) as f:
                terms = list(set(json.load(f)))  # 去重处理
            
            # 并行处理解释获取
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                explanations = list(tqdm(executor.map(self.get_explanation, terms),
                                       total=len(terms), desc=f"Fetching {json_file}",
                                       unit="term", leave=False))
            
            # 批量插入数据
            with tqdm(total=len(terms), desc=f"Inserting {json_file}", unit="term") as pbar:
                for i in range(0, len(terms), self.batch_size):
                    batch = list(zip(terms[i:i+self.batch_size], 
                                  explanations[i:i+self.batch_size]))
                    conn.executemany('''
                        INSERT OR IGNORE INTO terms (term, explanation)
                        VALUES (?, ?)
                    ''', batch)
                    pbar.update(len(batch))
                conn.commit()

    def process_files(self):
        for json_file, db_file in tqdm(self.file_map, desc="Total progress", unit="db"):
            self.process_file(json_file, db_file)

if __name__ == "__main__":
    # 根据API承受能力调整参数 (建议4-8 workers)
    builder = NounDatabaseBuilderOptimized(max_workers=6, batch_size=100)
    builder.process_files()