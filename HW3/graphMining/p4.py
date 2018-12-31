import community
import igraph
from igraph import *
import networkx as nx
import matplotlib.pyplot as plt

G = igraph.read('./karate/karate.gml')

# [m for m in dir(g) if m.startswith("write")]
# [m for m in dir(g) if m.startswith("layout")]
G.write_svg("graph.svg", layout=G.layout_kamada_kawai())

G.vs[0]["color"] = "red"
print(G.vs[0])

G.write_svg("graph2.svg", layout=G.layout_kamada_kawai())

# G = nx.path_graph(10)
# comp = girvan_newman(G)

# labels=list(G.vs['label'])
# N=len(labels)
# E=[e.tuple for e in G.es]# list of edges
# layt=G.layout('kk')

# plot(G)

# igraph_community_multilevel(G)
# part = community.best_partition(G)
# values = [part.get(node) for node in G.nodes()]

# nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = values, node_size=30, with_labels=False)

# first compute the best partition
# partition = community.best_partition(G)
#
# # drawing
# size = float(len(set(partition.values())))
# pos = nx.spring_layout(G)
# count = 0.
# for com in set(partition.values()):
#     count = count + 1.
#     list_nodes = [nodes for nodes in partition.keys()
#                   if partition[nodes] == com]
#     nx.draw_networkx_nodes(G, pos, list_nodes, node_size=20,
#                            node_color=str(count / size))
#
# nx.draw_networkx_edges(G, pos, alpha=0.5)
# plt.show()
