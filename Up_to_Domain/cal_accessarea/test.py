import networkx as nx

G = nx.MultiDiGraph()

G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)
G.add_node(5)
G.add_node(6)
G.add_node(7)
G.add_node(8)

G.add_edge(1, 2, weight=1)
G.add_edge(3, 1, weight=4)
G.add_edge(5, 3, weight=1)
G.add_edge(5, 4, weight=2)
G.add_edge(5, 6, weight=5)
G.add_edge(5, 8, weight=2)
G.add_edge(5, 7, weight=1)
G.add_edge(7, 8, weight=4)
G.add_edge(8, 6, weight=2)
G.add_edge(6, 4, weight=3)
G.add_edge(4, 2, weight=7)

paths = nx.shortest_path(G, 5, method="dijkstra", weight="weight")
print(paths)
