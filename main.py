from Graph import Graph
from time import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from mpi4py import MPI
import pymp

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

# PyPy
def get_neighbors(graph, vertex):
    neighbors = []

    for v, i in enumerate(graph.adj_matrix[vertex]):
        if i != 0 and (v not in neighbors): 
            neighbors.append(v)

    return neighbors

# PyPy
def vertex_degree(vertex, graph):
    degree = 0
    degree += (len(graph.adj_matrix[vertex]) - graph.adj_matrix[vertex].count(0))

    for i in range(0, graph.number_of_nodes):
        if graph.adj_matrix[i][vertex] != 0: degree += 1
        
    return degree

# PyPy
def max_weight(graph):
    max_degree = []

    for i in range(0, graph.number_of_nodes):
        max_degree.append(vertex_degree(i, graph))

    return max(max_degree)


def main():
    file_name = "web-polblogs.mtx"

    original_graph = None

    with open(file_name, 'r') as data:
        data.readline()

        (n, n, e) = data.readline().split(' ')
        original_graph = Graph(int(n))
        
        lines = data.readlines()

        # MPI para scatter
        for line in lines:
            values = line.replace('\n', '').split(' ')

            if len(values) == 3:
                original_graph.add_adj(v1=int(values[0]), v2=int(values[1]), weight=int(values[2]))
            else:
                original_graph.add_adj(v1=int(values[0]), v2=int(values[1]), weight=np.random.rand())
        
    prev_list = []

    start = time()
    with pymp.Parallel(8) as p:
        for i in p.range(0, int(n)):
            prev_list.append(dijsktra(original_graph, i))

        graph_list = []

        # Cria os subgrafos gerados
        for subgraph in p.iterate(prev_list):
            graph = Graph(int(n))

            for i, j in enumerate(subgraph):
                if j != None: graph.add_adj(v1=i, v2=j)
            graph_list.append(graph)

        tsp = None
    
        # Acha o grafo com grau maximo 2 e testa se existe uma aresta que conecta dos vertices
        # de grau impar
        for graph in graph_list:
            # p.print(graph)
            if max_weight(graph) == 2:
                nodes = []

                for i in range(0, graph.number_of_nodes):
                    if vertex_degree(i, graph) == 1:
                        nodes.append(i)
                
                if len(nodes) == 2:
                    v1, v2 = nodes

                    if original_graph.adj_matrix[v1][v2] != 0:
                        graph.adj_matrix[v1][v2] = 1
                        tsp = graph
                        break
                    elif original_graph.adj_matrix[v2][v1] != 0:
                        graph.adj_matrix[v2][v1] = 1
                        tsp = graph
                        break
    end = time()
        # Draw the result graph on the screen
        # if tsp != None:
        #     rows, cols = np.where(np.matrix(tsp.adj_matrix) == 1)
        #     edges = zip(rows.tolist(), cols.tolist())

        #     gr = nx.Graph()
        #     gr.add_edges_from(edges)
        #     nx.draw(gr, node_size=500)
        #     plt.show()
    print("Tempo de execução paralelo: %f" % (end - start))

if __name__ == "__main__":
    main()