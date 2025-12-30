import os
os.environ["PATH"] += os.pathsep + '/opt/homebrew/bin/'  # 确保Graphviz路径

import torch
from torchview import draw_graph
from transformers import AutoTokenizer
from train_entity import SarcasmModel, Config

# 1. 抑制预期内的警告
def suppress_expected_warnings():
    import logging
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

suppress_expected_warnings()

# 2. 加载模型（精简模式）
model = SarcasmModel.from_pretrained(
    "model/entity/reddit_model",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).eval()

# 3. 准备优化后的输入样本
def prepare_optimized_input():
    tokenizer = AutoTokenizer.from_pretrained(Config.text_model)
    
    # 最小化输入长度
    sample_text = "Short text example."  # 简化文本
    entities = ["example"] + [""]*(Config.max_entities-1)  # 单实体示例
    
    text_enc = tokenizer(
        sample_text,
        max_length=64,  # 缩短序列长度
        padding='max_length',
        return_tensors='pt'
    )
    
    entity_enc = tokenizer(
        entities,
        max_length=16,  # 缩短实体长度
        padding='max_length',
        return_tensors='pt'
    )
    
    return (
        text_enc.input_ids,
        text_enc.attention_mask,
        entity_enc.input_ids.view(1, Config.max_entities, -1),
        entity_enc.attention_mask.view(1, Config.max_entities, -1)
    )

# 4. 设备配置
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
inputs = [tensor.to(device) for tensor in prepare_optimized_input()]

# 5. 分层可视化配置
graph = draw_graph(
    model,
    input_data=inputs,
    depth=4,  # 控制展开深度
    hide_inner_tensors=True,
    hide_module_functions=True,
    roll=True,  # 合并重复结构
    expand_nested=True,  # 展开嵌套模块
    graph_name='SarcasmModel_Arch',
    directory='visualization',
    filename='optimized_arch',
    device=device
)

# 6. 生成矢量图
graph.visual_graph.format = 'svg'
graph.visual_graph.render(cleanup=True, view=False)

print("""
可视化生成成功！请查看：
├── visualization/optimized_arch.svg （矢量图，可无限缩放）
└── visualization/optimized_arch.gv  （原始Graphviz文件）
""")