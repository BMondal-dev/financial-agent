import json
import networkx as nx

METADATA_PATH = "../data/metadata/metadata.json"
GRAPH_PATH = "../data/stock_graph.json"

with open(METADATA_PATH) as f:
    metadata = json.load(f)

G = nx.Graph()

stocks = list(metadata.keys())

for stock in stocks:
    G.add_node(stock)

for s1 in stocks:
    for s2 in stocks:

        if s1 == s2:
            continue

        w = 0

        # correlation
        corr_list = metadata[s1]["top_correlated"]
        for c in corr_list:
            if c["symbol"] == s2:
                w += 0.4 * c["corr"]

        # sector similarity
        if metadata[s1]["sector"] == metadata[s2]["sector"]:
            w += 0.2

        # volatility similarity
        if metadata[s1]["volatility_bucket"] == metadata[s2]["volatility_bucket"]:
            w += 0.2

        # market cap similarity
        if metadata[s1]["market_cap_bucket"] == metadata[s2]["market_cap_bucket"]:
            w += 0.2

        if w > 0:
            G.add_edge(s1, s2, weight=w)

graph_data = nx.node_link_data(G)

with open(GRAPH_PATH, "w") as f:
    json.dump(graph_data, f, indent=2)

print("Stock similarity graph built.")