import explorer

e = explorer.WikidataExplorer()
item="BTC"
results = e.explore_term(item)
print(e.format_relation_graph(results, item))