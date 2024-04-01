import math

class WeightedGraph:

    def __init__(self,nodes):
        self.graph=[]
        self.weights={}
        for node in range(nodes):
            self.graph.append([])

    def add_node(self,node):
        self.graph[node]=[]

    def add_edge(self, node1, node2, weight):
        if node2 not in self.graph[node1]:
            self.graph[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def get_weights(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def are_connected(self, node1, node2):
        for neighbour in self.graph[node1]:
            if neighbour == node2:
                return True
        return False

    def get_neighbors(self, node):
        return self.graph[node]

    def get_number_of_nodes(self,):
        return len(self.graph)
    
    def get_nodes(self,):
        return [i for i in range(len(self.graph))]
    
    def get_number_of_edges(self):
        return sum(len(edges) for edges in self.graph)
    
def allpair(g):
    INF = math.inf
    n = g.get_number_of_nodes()
    m = g.get_number_of_edges()
    dist = [[INF for _ in range(n)] for _ in range(n)]
    prev = [[None for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i==j:
                dist[i][j]=0
    
    for i in range(n):
        edge = g.get_neighbors(i)
        if edge:
            for j in edge:
                dist[i][j] = g.get_weights(i,j)

    for visit in range(0,n):
        for source in range(0,n):
            for dest in range(0,n):
                if dist[source][visit] + dist[visit][dest] < dist[source][dest]:
                    dist[source][dest] = dist[source][visit] + dist[visit][dest]
                    prev[source][dest] = visit

    return dist, prev

    
g= WeightedGraph(6)

g.add_edge(0,1,15)
g.add_edge(1,0,15)

g.add_edge(1,2,10)
g.add_edge(2,1,10)

g.add_edge(2,3,5)
g.add_edge(3,2,5)

g.add_edge(1,3,20)
g.add_edge(3,1,20)

g.add_edge(2,4,30)
g.add_edge(4,2,30)

g.add_edge(4,3,25)
g.add_edge(3,4,25)

g.add_edge(3,5,40)
g.add_edge(5,3,40)

g.add_edge(4,5,15)

dist, prev = allpair(g)

print("Distance for all pairs: \n")
for row in dist:
    print(row)
# print(dist[5][0])
# print(prev[5][0])

print("\n \n")

print("Second-to-last vertex:")
for row in prev:
    print(row)