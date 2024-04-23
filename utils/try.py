import torch
import networkx as nx
print(torch.__version__)
import networkx as nx
import matplotlib.pyplot as plt
#
# # 创建一个无向图
# G = nx.Graph()
#
# # 添加节点和边
# G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (3, 5), (2, 4)])
#
# # # 可视化原始图
# # plt.figure(figsize=(5,5))
# # nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
# # plt.title("Original Graph")
# # plt.show()
#
# selected_nodes = [1,  4]
#
# # 从原始图中创建子图
# # subgraph = G.subgraph(selected_nodes).copy()
# connected_components = list(nx.connected_components(G))
# for component in connected_components:
#     if 1 in component or 4 in component:
#         subgraph_nodes = component
#         break
# subgraph = G.subgraph(subgraph_nodes)
# # 可视化子图
# plt.figure(figsize=(5,5))
# nx.draw(subgraph, with_labels=True, node_color='lightgreen', edge_color='gray')
# plt.title("Subgraph")
# plt.show()
# 创建一个多重边无向图
G = nx.MultiGraph()

# 添加一些节点和边
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3), (2, 4), (1, 5), (5, 6), (6, 7), (7, 5)])

# 感兴趣的节点列表
interested_nodes = [6,2]

# 创建一个空的子图
subG = nx.MultiGraph()

# 对于interested_nodes中的每一对节点，找到所有最短路径并添加到subG中
for i in range(len(interested_nodes)):
    for j in range(i + 1, len(interested_nodes)):
        source = interested_nodes[i]
        target = interested_nodes[j]
        # 对于每一对节点，找到所有最短路径
        for path in nx.all_shortest_paths(G, source=source, target=target):
            # 将路径上的边添加到子图中
            nx.add_path(subG, path)
for edge in subG.edges():
    print(edge)
# 如果你想可视化结果，可以使用以下代码
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title('graph')
plt.show()
plt.figure(figsize=(8, 4))
nx.draw(subG, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title('Subgraph with Indirectly Connected Nodes')
plt.show()