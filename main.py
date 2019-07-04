from Graph import Graph
from time import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpi4py import MPI
import pymp
import sys

#funcao dijkstra - encontra a arvore de caminho minimo entre todos os vértices a partir de um vertice inicial
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

# retorna os vizinhos de vertex
def get_neighbors(graph, vertex):
    neighbors = []

    for v, i in enumerate(graph.adj_matrix[vertex]):
        if i != 0 and (v not in neighbors): 
            neighbors.append(v)

    return neighbors

# retorna o grau do vertice vertex
def vertex_degree(vertex, graph):
    degree = 0
    degree += (len(graph.adj_matrix[vertex]) - graph.adj_matrix[vertex].count(0))

    for i in range(0, graph.number_of_nodes):
        if graph.adj_matrix[i][vertex] != 0: degree += 1
        
    return degree

# retorna o grau maximo do grafo
def max_weight(graph):
    max_degree = []
    for i in range(0, graph.number_of_nodes):
        max_degree.append(vertex_degree(i, graph))
    return max(max_degree)
    

#funcao principal
def main():
    #instancia o mpi
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    #custo_min = np.zeros(1, dtype=int)
    custo_min = None
    custoParcial = np.zeros(1, dtype=int)
    custoParcial[0] = int(999999999)
    tempoMax = 0.0
    #processo 0 le o grafo do arquivo e o envia para os demais processos
    if(rank == 0):
        #file_name = "web-polblogs.mtx"
        file_name = "web-polblogs-solucao.mtx"

        original_graph = None

        with open(file_name, 'r') as data:
            data.readline()

            (n, n, e) = data.readline().split(' ')
            original_graph = Graph(int(n))
            
            lines = data.readlines()

            #para cada linha do arquivo
            for line in lines:
                #values recebe uma lista que representa a aresta do modo: vertice vertice peso
                values = line.replace('\n', '').split(' ')
                #adiciona as arestas na matriz de adjacencia do objeto original_graph
                if len(values) == 3:
                    original_graph.add_adj(v1=int(values[0]), v2=int(values[1]), weight=int(values[2]))
                #se o grafo nao e' ponderado, valores das arestas e' randomico
                else:
                    original_graph.add_adj(v1=int(values[0]), v2=int(values[1]), weight=np.random.rand())
            
    #execucao dos demais processos  
    else:
        original_graph = None
    #todos os processos executam o trecho de codigo abaixo

    start = time()

    #enviar o grafo para os processos
    original_graph = comm.bcast(original_graph, root=0)
    
    #calculo o tamanho do intervalo
    intervalo_v = int(original_graph.number_of_nodes/size)
    #prev_list ira armazenar as arvores de caminho minimo calculadas
    #cada processo ira encontrar arvores de caminho minimo de acordo com o intervalo dedicado a ele 
    prev_list = []
    #todas exceto o ultimo processo
    if(rank < (size-1)):
        #processo calcula as arvores de caminho minimo destinadas a si
        for i in range(rank*intervalo_v, (rank+1)*intervalo_v):
            prev_list.append(dijsktra(original_graph, i))
    if(rank == size-1):
        #ultimo processo calculo tambem o restante do intervalo, caso o resto da divisao num vertices / size for maior que 0
        resto = original_graph.number_of_nodes - size * intervalo_v
        for i in range(rank*intervalo_v+1, (rank+1)*intervalo_v+resto):
            prev_list.append(dijsktra(original_graph, i))

    
    graph_list = []
    
    # Cria os subgrafos gerados
    for prev in prev_list:
        subgraph = prev
        graph = Graph(original_graph.number_of_nodes)

        for i, j in enumerate(subgraph):
            if j != None: graph.add_adj(v1=i, v2=j)
        graph_list.append(graph)

    #tsp guarda uma solução para o caixeiro viajante
    tsp = []
    # Acha o grafo com grau maximo 2 e testa se existe uma aresta que conecta dos vertices de grau impar
    for graph in graph_list:
        #se o grau maximo do grafo e' 2, entao pode ser que tenha uma solucao
        if max_weight(graph) == 2:
            nodes = []
            #encontra se ha somente dois vertices de grau 1 no subgrafo
            #se houver, verifica se no grafo original pode-se conectar ambos os vertices, se sim temos uma solucao

            for i in range(0, graph.number_of_nodes):
                if vertex_degree(i, graph) == 1:
                    nodes.append(i)
            
            if len(nodes) == 2:
                v1, v2 = nodes

                if original_graph.adj_matrix[v1][v2] != 0:
                    graph.adj_matrix[v1][v2] = 1
                    tsp.append(graph)
                elif original_graph.adj_matrix[v2][v1] != 0:
                    graph.adj_matrix[v2][v1] = 1
                    tsp.append(graph)
    #encontrar o tsp de menor custo
    tspSolucao = None
    #se há solucoes, vamos verificar a de menor custo
    if tsp != []:
        soma = 0
        for graph in tsp:
            for i in range(0,len(graph.adj_matrix)):
                for j in range(0,len(graph.adj_matrix[0])):
                    soma+= graph.adj_matrix[i][j]
            if(soma < custoParcial):
                tspSolucao = graph
                custoParcial = int(soma)
    #reduce para encontrar a menor solucao de cada processo e retornar para o processo 0
    custo_min = comm.reduce(custoParcial, op=MPI.MIN,root=0)

    end = time()
    tempoMax = end - start
    tempo = comm.reduce(tempoMax,op=MPI.MAX, root=0 )
    #Draw the result graph on the screen
    
    if tspSolucao != None:
        print("plotando")
        rows, cols = np.where(np.matrix(tspSolucao.adj_matrix) == 1)
        edges = zip(rows.tolist(), cols.tolist())

        gr = nx.Graph()
        gr.add_edges_from(edges)
        nx.draw(gr, node_size=300, with_labels=True, font_weight='bold')
        plt.show()
    

    if (rank == 0):
        if custo_min != 999999999:
            print("menor custo",custo_min)
        else:
            print('Grafo sem solução!')
        print("tempo máximo :", tempo)

if __name__ == "__main__":
    main()