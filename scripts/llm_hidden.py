from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
assert torch.backends.mps.is_available()
device = torch.device("mps")

# 修正1：使用正确的模型加载方式
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.mps.is_available() else torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

# 修正2：正确的旋转参数访问方式
with torch.no_grad():
    # Phi-3 使用隐藏的_RotaryEmbedding类
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "qkv_proj"):
            layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)
        # DEBUG: 检查所有参数设备位置
        for name, param in layer.named_parameters():
            if param.device != device:
                param.data = param.data.to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def analyze_sarcasm(text: str) -> tuple:
    prompt = f"<|user|>\nAnalyze sarcasm in: '{text}'<|end|>\n<|assistant|>"
    # prompt = f"{text}"
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=256,
        truncation=True
    ).to(device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.9,
            return_dict_in_generate=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 修正隐藏状态提取逻辑
    # outputs.hidden_states 结构: (layers) x (tokens) x (batch, seq_len, dim)
    last_layer_hidden = outputs.hidden_states[-1]    # 获取最后一层所有token的隐藏状态
    hidden_state = last_layer_hidden[-1][0, -1].cpu()  # 提取最后一个生成token的特征
    
    response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    answer = "yes" if any(kw in response.lower() for kw in ["yes", "sarcastic"]) else "no"
    
    return hidden_state, response, answer

if __name__ == "__main__":
    test_texts = [
        "you are good.",
        "you are stupid."
    ]
    
    for text in test_texts:
        vec, resp, ans = analyze_sarcasm(text)
        print(f"\nInput: {text}")
        print(f"Response: {resp.split('<|assistant|>')[-1].strip()}")
        print(f"Answer: {ans}")
        print(f"Vector Shape: {vec.shape} (Norm: {torch.norm(vec):.2f})") # 1x3072
        print(f"vector {vec}")
        print("="*60)

    # 对比不同输入的向量差异

    
    # text1 = "you are good"
    # vec1, _, _ = analyze_sarcasm(text1)

    # text2 = "you are stupid"
    # vec2, _, _ = analyze_sarcasm(text2)

    # # 计算余弦相似度
    # cos_sim = torch.cosine_similarity(vec1, vec2, dim=0)
    # print(f"相似度: {cos_sim:.2f}")