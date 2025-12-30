from conceptnet_lite import connect, Label, Edge
from functools import reduce
from peewee import *

connect()

def visualize_concept_graph(word, lang='en', limit=20):
    try:
        label = Label.get(text=word, language=lang)
        if not label.concepts:
            print(f"未找到与 '{word}' 相关的概念")
            return
        
        # 选择关系最多的概念
        concept = max(
            label.concepts,
            key=lambda c: Edge.select().where(Edge.start == c).count()
        )
        
        # 扩展关系类型列表
        irony_related_relations = [
            'antonym', 'related_to', 'synonym', 'is_a',
            'has_a', 'part_of', 'used_for', 'causes',
            'has_property', 'at_location', 'form_of'
        ]
        
        # 构建查询条件
        conditions = [Edge.relation.name == rel for rel in irony_related_relations]
        combined_condition = reduce(lambda a, b: a | b, conditions)
        
        query = (
            Edge.select()
            .where((Edge.start == concept) & combined_condition)
            .limit(limit)
        )
        
        print(f"中心概念: {concept.uri} ({concept.label.text})")
        if not query:
            print("⚠️ 未找到符合条件的关系（尝试扩展关系类型列表）")
            return
            
        for idx, edge in enumerate(query, 1):
            end_concept = edge.end
            print(f"关系 {idx}: {edge.relation.name}")
            print(f"  起点: {concept.label.text}")
            print(f"  终点: {end_concept.label.text}")
            print(f"  边URI: {edge.uri}")
            print("-" * 50)
            
    except Exception as e:
        print(f"查询失败: {e}")

# 测试高覆盖率词汇
test_words = ['apple']
for word in test_words:
    print(f"\n测试词: {word}")
    visualize_concept_graph(word)
