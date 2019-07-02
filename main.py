from Graph import Graph
from time import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def get_neighbors(graph, vertex):
    neighbors = []

    for v, i in enumerate(graph.adj_matrix[vertex]):
        if i != 0 and (v not in neighbors): 
            neighbors.append(v)

    return neighbors

def dijsktra(graph, initial):
    dist = []
    prev = []

    for i in range(0, graph.number_of_nodes):
        dist.append(float("inf"))
        prev.append(None)
    
    dist[initial] = 0
    Q = list(range(0, graph.number_of_nodes))

    while len(Q) != 0:
        u_dist = float('inf')
        u_index = Q[0]

        # Returns the vertex with the smalest distance from the source
        for (i, d) in enumerate(dist):
            if i in Q and d < u_dist:
                u_dist = d
                u_index = i

        Q.remove(u_index)

        for v in get_neighbors(graph, u_index):
            alt = dist[u_index] + graph.adj_matrix[u_index][v]

            # Relax U and V
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u_index
    return prev

def vertex_degree(vertex, graph):
    degree = 0
    degree += (len(graph.adj_matrix[vertex]) - graph.adj_matrix[vertex].count(0))

    for i in range(0, graph.number_of_nodes):
        if graph.adj_matrix[i][vertex] != 0: degree += 1
        
    return degree


def max_weight(graph):
    max_degree = []

    for i in range(0, graph.number_of_nodes):
        max_degree.append(vertex_degree(i, graph))

    return max(max_degree)


# Main scope
file_name = "web-polblogs.mtx"

original_graph = None

with open(file_name, 'r') as data:
    data.readline()

    (n, n, e) = data.readline().split(' ')
    original_graph = Graph(int(n))
    
    lines = data.readlines()

    for line in lines:
        values = line.replace('\n', '').split(' ')

        if len(values) == 3:
            original_graph.add_adj(v1=int(values[0]), v2=int(values[1]), weight=int(values[2]))
        else:
            original_graph.add_adj(v1=int(values[0]), v2=int(values[1]))
    
prev_list = []

start = time()
for i in range(0, int(n)):
    prev_list.append(dijsktra(original_graph, i))
end = time()

graph_list = []

for subgraph in prev_list:
    graph = Graph(int(n))

    for i, j in enumerate(subgraph):
        if j != None: graph.add_adj(v1=i, v2=j)
    graph_list.append(graph)

g = 0

for i, graph in enumerate(graph_list):
    if max_weight(graph) <= 2:
        g = i
        nodes = []

        for i in range(0, graph.number_of_nodes):
            if vertex_degree(i, graph) == 1:
                nodes.append(i)
        
        if len(nodes) == 2:
            v1, v2 = nodes

            if original_graph.adj_matrix[v1][v2] != 0:
                graph.adj_matrix[v1][v2] = 1
                break
            elif original_graph.adj_matrix[v2][v1] != 0:
                graph.adj_matrix[v2][v1] = 1
                break

rows, cols = np.where(np.matrix(graph_list[g].adj_matrix) == 1)
edges = zip(rows.tolist(), cols.tolist())

gr = nx.Graph()
gr.add_edges_from(edges)
nx.draw(gr, node_size=500)
plt.show()

print("Tempo de execução: %f" % (end - start))

