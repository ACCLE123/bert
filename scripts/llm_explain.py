# llm_integration.py
import requests
import time
from typing import Optional

class LLMExplanationGenerator:
    """
    LLM解释生成器，支持本地Ollama API调用
    功能特性：
    - 自动重试机制（指数退避）
    - 请求缓存优化
    - 响应格式验证
    - 性能监控
    """
    
    def __init__(self, model_name: str = "llama3", timeout: int = 45):
        self.base_url = "http://localhost:11434/api/generate"
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = 5
        self.cache = {}
        self.stats = {
            "total_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0
        }

    def _build_prompt(self, sentence: str, entity: str) -> str:
        """构建结构化提示模板"""
        return f"""
        Analyze the sarcastic usage of entity in context. Follow this structure:
        
        [Text]
        {sentence}
        
        [Entity]
        {entity}
        
        [Analysis Requirements]
        1. Literal meaning vs Contextual contradiction
        2. Cultural/subcultural implications
        3. Sentiment polarity reversal evidence
        
        [Response Format]
        Use bullet points with clear numbering. Avoid markdown formatting.
        """

    def generate(self, sentence: str, entity: str) -> Optional[str]:
        """
        生成实体解释（主入口方法）
        参数：
            sentence: 原始文本（需预先清洗）
            entity: 目标实体
        返回：
            结构化解释文本 或 None
        """
        cache_key = f"{sentence[:150]}|{entity}"  # 限制key长度
        
        # 缓存检查
        if cached := self.cache.get(cache_key):
            return cached

        prompt = self._build_prompt(sentence, entity)
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.5, "top_p": 0.9}
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()

                result = response.json()
                explanation = self._validate_output(result["response"])
                
                # 更新统计
                self._update_stats(start_time, success=True)
                self.cache[cache_key] = explanation
                return explanation

            except (requests.RequestException, KeyError) as e:
                delay = 2 ** attempt  # 指数退避
                print(f"Attempt {attempt+1} failed: {str(e)}. Retrying in {delay}s...")
                time.sleep(delay)
                self._update_stats(start_time, success=False)

        print(f"Failed after {self.max_retries} attempts")
        return None

    def _validate_output(self, text: str) -> Optional[str]:
        """验证响应格式"""
        required_keywords = ["literal", "context", "cultural", "sentiment"]
        text_lower = text.lower()
        
        # 检查关键要素
        if all(kw in text_lower for kw in required_keywords):
            # 标准化格式
            return "\n".join([line.strip() for line in text.split("\n") if line.strip()])
        return None

    def _update_stats(self, start_time: float, success: bool):
        """更新性能统计"""
        duration = time.time() - start_time
        self.stats["total_requests"] += 1
        self.stats["avg_response_time"] = (
            self.stats["avg_response_time"] * (self.stats["total_requests"] - 1) + duration
        ) / self.stats["total_requests"]
        
        if not success:
            self.stats["failed_requests"] += 1

    def get_performance(self) -> dict:
        """获取性能指标"""
        return {
            "success_rate": 1 - (self.stats["failed_requests"] / self.stats["total_requests"]),
            "avg_response_sec": round(self.stats["avg_response_time"], 2),
            "cache_size": len(self.cache)
        }

# 使用示例
if __name__ == "__main__":
    llm = LLMExplanationGenerator()
    
    test_cases = [
        ("This 'state-of-the-art' system crashed again", "state-of-the-art"),
        ("Great job missing all the deadlines!", "Great job"),
        ("Another 'urgent' email that could've been a text", "urgent")
    ]
    
    for text, entity in test_cases:
        explanation = llm.generate(text, entity)
        print(f"Input: {entity}\nExplanation:\n{explanation}\n{'-'*40}")
    
    print("Performance Metrics:", llm.get_performance())