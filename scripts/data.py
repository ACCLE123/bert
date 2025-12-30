# API_KEY = "sk-4dfb5971240741f89df2353525837de4"

import csv
import re
import requests
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Set

# 配置参数
CSV_PATH = "hybrid_data.csv"
API_KEY = "sk-4dfb5971240741f89df2353525837de4"  # 替换真实API密钥
BATCH_SIZE = 120                  # 每批生成数量
TARGET_COUNT = 10000
MAX_WORKERS = 3                    # 并发线程数
REQUEST_TIMEOUT = 45               # 请求超时时间(秒)
VALID_RELATIONS = {'antonym', 'not_has_property', 'not_used_for',
                  'causes', 'has_context', 'entails', 'obstructed_by'}

# 全局状态
SEEN_KEYS: Set[str] = set()
LOCK = threading.Lock()
REQUEST_HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def enhanced_parser(raw_text: str) -> List[Tuple]:
    """工业级数据解析器，处理多种异常格式"""
    parsed = []
    # 清除所有非数据内容
    clean_text = re.sub(
        r'(```[\s\S]*?```|^#.*|Examples?:.*|Note:.*|\b\w+:\s*|\*+)',
        '',
        raw_text,
        flags=re.MULTILINE
    )
    
    for line in clean_text.split('\n'):
        line = line.strip()
        if not line or len(line) < 10:  # 过滤短行
            continue
        
        # 处理多种分隔符和格式变体
        line = re.sub(r'\s{2,}', ',', line)  # 替换多个空格为逗号
        if '","' in line:  # 处理带引号的CSV
            parts = [p.strip('"') for p in line.split('","')]
        else:
            parts = re.split(r'[,|]\s*', line, maxsplit=3)
            
        if len(parts) != 4:
            continue
            
        try:
            # 数据标准化和验证
            source = parts[0].strip().replace(' ', '_').lower()
            relation = parts[1].strip().lower()
            target = parts[2].strip().replace(' ', '_').lower()
            weight = round(float(parts[3].strip()), 1)
            
            if (relation not in VALID_RELATIONS or 
                not (0.5 <= weight <= 2.5) or
                len(source) < 2 or len(target) < 2):
                continue
                
            parsed.append((source, relation, target, weight))
        except (ValueError, IndexError):
            continue
            
    return parsed

def api_request_with_retry(prompt: str) -> List[Tuple]:
    """带智能退避的API请求"""
    for attempt in range(5):
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [{
                        "role": "user",
                        "content": prompt + "\n重要! 必须严格遵守以下格式:\n1. 严格使用CSV格式\n2. 不要包含任何注释\n3. 不要使用Markdown"
                    }],
                    "temperature": 0.3,
                    "max_tokens": 3000
                },
                headers=REQUEST_HEADERS,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return enhanced_parser(response.json()['choices'][0]['message']['content'])
            
        except requests.exceptions.RequestException as e:
            wait = (2 ** attempt) + random.uniform(0, 2)
            print(f"请求失败: {e}，等待{wait:.1f}秒后重试...")
            time.sleep(wait)
        except Exception as e:
            print(f"意外错误: {e}")
            time.sleep(5)
            
    return []

def generate_hybrid_prompt(batch_num: int) -> str:
    """优化后的混合数据提示词"""
    return f"""Generate {BATCH_SIZE} hybrid knowledge relations with:

Required domain mix:
- 40% Internet slang (HODL, FOMO, mooning)
- 30% Financial terms (short_selling, dark_pool)
- 20% Social events (GameStop_squeeze)
- 10% Sensitive topics (racial_profiling)

Rules:
1. Strict CSV format: source,relation,target,weight
2. Relations: {', '.join(VALID_RELATIONS)}
3. Weights: 0.5-2.5 (1 decimal)
4. No markdown/headers/explanation
5. Unique combinations
6. Batch ID: {batch_num}
"""

def save_batch(data: List[Tuple]):
    """线程安全存储"""
    with LOCK:
        current_count = len(SEEN_KEYS)
        if current_count >= TARGET_COUNT:
            return
            
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            new_count = 0
            
            for source, rel, target, weight in data:
                key = f"{source}|{rel}|{target}"
                if key not in SEEN_KEYS:
                    SEEN_KEYS.add(key)
                    writer.writerow([source, rel, target, weight])
                    new_count += 1
                    
                    if len(SEEN_KEYS) >= TARGET_COUNT:
                        break
                        
            if new_count > 0:
                print(f"保存 {new_count} 条数据 | 总计: {len(SEEN_KEYS)}/{TARGET_COUNT}")

def worker(batch_num: int):
    """增强型工作线程"""
    print(f"启动批次 {batch_num}")
    start_time = time.time()
    
    try:
        prompt = generate_hybrid_prompt(batch_num)
        data = api_request_with_retry(prompt)
        
        if data:
            save_batch(data)
            time_cost = time.time() - start_time
            print(f"完成批次 {batch_num} | 耗时: {time_cost:.1f}s | 新增: {len(data)}条")
        else:
            print(f"批次 {batch_num} 无有效数据")
            
    except Exception as e:
        print(f"批次 {batch_num} 失败: {e}")

def main():
    # 初始化CSV文件
    with open(CSV_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(["source_concept", "relation_type", "target_concept", "weight"])
    
    # 动态批次控制
    batch_num = 1
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        
        while len(SEEN_KEYS) < TARGET_COUNT:
            # 动态控制并发
            active_workers = sum(not f.done() for f in futures)
            if active_workers < MAX_WORKERS:
                future = executor.submit(worker, batch_num)
                futures.append(future)
                batch_num += 1
                time.sleep(random.uniform(0.5, 2))  # 随机间隔
            else:
                time.sleep(1)
                
            # 清理完成的任务
            futures = [f for f in futures if not f.done()]
            
    print(f"生成完成! 总计 {len(SEEN_KEYS)} 条数据")

if __name__ == "__main__":
    main()