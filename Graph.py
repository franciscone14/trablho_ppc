class Graph:
    def __init__(self, n):
        self.number_of_nodes = n
        self.adj_matrix = []

        for i in range(0,n):
            line = []
            for j in range(0,n):
                line.append(0)
            self.adj_matrix.append(line)

    def add_adj(self, v1, v2, weight = 1):
        self.adj_matrix[v1 - 1][v2 - 1] = weight
