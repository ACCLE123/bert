from transformers import AutoConfig

config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
print("可用配置参数:", dir(config))