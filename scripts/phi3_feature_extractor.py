# phi3_feature_extractor.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Tuple, List
import os

# 禁用动态图优化
os.environ["TORCHDYNAMO_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True

class Phi3FeatureExtractor:
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.device = torch.device("mps")
        self.model_name = model_name
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self._prepare_model()
    
    def _load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """加载模型和分词器"""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # 强制使用fp16
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer
    
    def _prepare_model(self):
        """模型准备步骤"""
        with torch.no_grad():
            # 确保所有参数在MPS设备
            for param in self.model.parameters():
                param.data = param.data.to(self.device)

    def _create_prompt(self, text: str) -> str:
        return f"<|user|>\nAnalyze text: '{text}'<|end|>\n<|assistant|>"
    
    @torch.inference_mode()
    def extract_features(self, text: str) -> torch.Tensor:
        """单文本特征提取"""
        prompt = self._create_prompt(text)
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=4096, 
            truncation=True
        ).to(self.device)
        
        outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        return last_hidden[0, -1].float()
    
    @torch.inference_mode()
    def batch_extract(self, texts: List[str]) -> torch.Tensor:
        """批量特征提取优化版"""
        prompts = [self._create_prompt(t) for t in texts]
        
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt"
        ).to(self.device)

        # 手动计算序列长度
        seq_lengths = inputs.attention_mask.sum(dim=1) - 1
        
        # 单次前向传播
        outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        
        # 高效索引最后token
        features = last_hidden[torch.arange(len(last_hidden)), seq_lengths]
        return features.float().to(self.device)
    

# 在文件末尾添加以下测试代码
if __name__ == "__main__":
    # 测试配置
    test_texts = [
        "This is absolutely fantastic!",    # 正面文本
        "Oh great, another system crash.",  # 讽刺文本
        "The performance is lightning fast"  # 中性文本
    ]
    
    def run_tests():
        """运行完整测试流程"""
        try:
            # 初始化特征提取器
            print("="*50)
            print("初始化Phi-3特征提取器...")
            extractor = Phi3FeatureExtractor()
            
            # 验证设备配置
            print(f"\n设备验证:")
            print(f"模型设备: {next(extractor.model.parameters()).device}")
            print(f"测试张量设备: {torch.tensor([0]).to(extractor.device).device}")
            
            # 测试单样本处理
            print("\n" + "="*50)
            print("单文本处理测试:")
            single_feature = extractor.extract_features(test_texts[0])
            print(f"特征形状: {single_feature.shape}")
            print(f"特征数据类型: {single_feature.dtype}")
            print(f"特征设备: {single_feature.device}")
            print(f"特征值范围: [{single_feature.min():.4f}, {single_feature.max():.4f}]")
            
            # 测试批量处理
            print("\n" + "="*50)
            print("批量处理测试:")
            batch_features = extractor.batch_extract(test_texts)
            print(f"批量特征形状: {batch_features.shape}")
            print(f"样本间余弦相似度矩阵:")
            for i in range(len(test_texts)):
                for j in range(len(test_texts)):
                    sim = torch.cosine_similarity(
                        batch_features[i].unsqueeze(0),
                        batch_features[j].unsqueeze(0)
                    ).item()
                    print(f"Text{i+1} vs Text{j+1}: {sim:.4f}", end=" | ")
                print()
            
            # 测试极端情况
            print("\n" + "="*50)
            print("边界条件测试:")
            empty_feature = extractor.extract_features("")
            print(f"空文本特征形状: {empty_feature.shape}")
            
            long_text = " ".join(["test"] * 1000)
            long_feature = extractor.extract_features(long_text)
            print(f"长文本特征形状: {long_feature.shape}")
            
            print("\n所有测试通过！")
            
        except Exception as e:
            print(f"\n测试失败: {str(e)}")
            import traceback
            traceback.print_exc()

    # 执行测试
    run_tests()