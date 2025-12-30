from SPARQLWrapper import SPARQLWrapper, JSON
import time

class WikidataEntity:
    def __init__(self, qid, label, description, relations, importance_score):
        self.qid = qid
        self.label = label
        self.description = description
        self.relations = relations  # 改为包含权重的列表结构
        self.importance_score = importance_score

class WikidataExplorer:
    def __init__(self):
        self.sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(60)
        
        # 配置属性权重字典
        self.relation_config = {
            "分类体系": {
                "P31": {"label": "实例类型", "weight": 0.8},
                "P279": {"label": "上级分类", "weight": 0.7}
            },
            "核心关联": {
                "P527": {"label": "组成部分", "weight": 0.6},
                "P361": {"label": "所属系列", "weight": 0.5}
            }
        }


    def _build_query(self, term):
        select_clause = ["?entity", "?entityLabel"]
        where_clause = []
        used_vars = set()
        
        # 生成查询变量
        for category, props in self.relation_config.items():
            for prop, config in props.items():
                var_base = prop.lower()
                var = var_base
                suffix = 1
                
                # 生成唯一变量名
                while var in used_vars:
                    var = f"{var_base}_{suffix}"
                    suffix += 1
                used_vars.add(var)
                
                select_clause.append(
                    f"(GROUP_CONCAT(DISTINCT ?{var}Label; SEPARATOR=' | ') AS ?{var}s)"
                )
                where_clause.append(f"""
                    OPTIONAL {{
                        ?entity wdt:{prop} ?{var}.
                        ?{var} rdfs:label ?{var}Label.
                        FILTER(LANG(?{var}Label) = "en")
                    }}
                """)

        return f"""
        SELECT {" ".join(select_clause)}
               (SAMPLE(?desc) AS ?description)
        WHERE {{
            SERVICE wikibase:mwapi {{
                bd:serviceParam wikibase:api "EntitySearch".
                bd:serviceParam wikibase:endpoint "www.wikidata.org".
                bd:serviceParam mwapi:search "{term}".
                bd:serviceParam mwapi:language "en".
                ?entity wikibase:apiOutputItem mwapi:item.
            }}
            # 匹配标签和别名
            ?entity rdfs:label|skos:altLabel ?label.
            FILTER(LANG(?label) = "en").
            {" ".join(where_clause)}
            OPTIONAL {{
                ?entity schema:description ?desc.
                FILTER(LANG(?desc) = "en")
            }}
            SERVICE wikibase:label {{
                bd:serviceParam wikibase:language "en".
            }}
        }}
        GROUP BY ?entity ?entityLabel
        ORDER BY DESC(STR(?label) = "{term}")  # 优先完全匹配的标签
               DESC(CONTAINS(LCASE(?label), LCASE("{term}")))  # 优先部分匹配的标签
               DESC(CONTAINS(LCASE(?desc), LCASE("{term}")))  # 优先描述中包含查询词的实体
        LIMIT 10
        """


    def explore_term(self, term, max_retries=5):
        """执行查询并返回结构化结果"""
        query = self._build_query(term)
        self.sparql.setQuery(query)
        
        retries = 0
        while retries < max_retries:
            try:
                results = self.sparql.query().convert()
                return self._parse_results(results)
            except Exception as e:
                retries += 1
                time.sleep(3)
        return []


    def _parse_results(self, results):
        entities = []
        # 创建属性映射字典 {p31s: ("实例类型", 0.8), ...}
        prop_mapping = {}
        for category in self.relation_config.values():
            for prop, config in category.items():
                key = prop.lower() + "s"
                prop_mapping[key] = (config["label"], config["weight"])

        for item in results["results"]["bindings"]:
            qid = item["entity"]["value"].split("/")[-1]
            label = item["entityLabel"]["value"]
            desc = item.get("description", {}).get("value", "")
            
            relations = []
            importance_score = 0.0
            
            # 解析每个属性关系
            for var_key in prop_mapping:
                if var_key in item:
                    rel_type, weight = prop_mapping[var_key]
                    values = item[var_key]["value"].split(" | ") if item[var_key]["value"] else []
                    
                    for val in values:
                        relations.append({
                            "relation": rel_type,
                            "value": val.strip(),
                            "weight": weight
                        })
                        # 累计重要性分数
                        importance_score += weight
            
            entities.append(WikidataEntity(
                qid=qid,
                label=label,
                description=desc,
                relations=sorted(relations, key=lambda x: x["weight"], reverse=True),
                importance_score=round(importance_score, 2)
            ))
        return entities

    def format_relation_graph(self, entities, term):
        """极简关系图谱"""
        output = []
        for entity in entities:
            output.append(f"{term}, {entity.description}")
            # print(entity.description)
            # for rel in entity.relations[:5]:
                # output.append(f"{term}, {entity.label}, {entity.description}, {rel['value']}")
        # for entity in entities:
        #     # 只保留实体名称和关联词
        #     relations = [rel['value'] for rel in entity.relations]
        #     output.append(
        #         f"{term} → {entity.label}\n" +
        #         f"关联词: {', '.join(relations)}"
        #     )
        return "\n".join(output)

# 使用示例
if __name__ == '__main__':
    explorer = WikidataExplorer()
    item="BTC"
    results = explorer.explore_term(item)
    
    # 获取关系图谱输出
    print(explorer.format_relation_graph(results, item))