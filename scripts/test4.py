import sqlite3
from pyvis.network import Network

conn = sqlite3.connect('data/nouns/reddit_train.db')
cursor = conn.cursor()

# 创建网络图
net = Network(height="600px", width="100%", notebook=True)

# 添加节点和边(这里简单展示术语)
cursor.execute("SELECT term, explanation FROM terms LIMIT 1000")  # 限制数量避免过于拥挤
for term, explanation in cursor.fetchall():
    net.add_node(term, title=explanation, shape="box")
    
# 显示图形
net.show("reddit_terms.html")
conn.close()