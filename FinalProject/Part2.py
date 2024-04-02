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
    

def dijkstra(g):
    INF = math.inf
    n = g.get_number_of_nodes()
    dist_result = [[INF for _ in range(n)] for _ in range(n)]
    prev = [[None for _ in range(n)] for _ in range(n)]

    for source in range(n):
        dist = [INF] * n
        dist[source] = 0
        visited = [False] * n

        while True:
            min_distance = INF
            min_node = -1
            for node in range(n):
                if not visited[node] and dist[node] < min_distance:
                    min_distance = dist[node]
                    min_node = node

            if min_node == -1:
                break

            visited[min_node] = True

            for neighbor in g.get_neighbors(min_node):
                distance = dist[min_node] + g.get_weights(min_node, neighbor)
                if distance < dist[neighbor]:
                    dist[neighbor] = distance
                    prev[source][neighbor] = min_node

        for dst in range(n):
            dist_result[source][dst] = dist[dst]

    return dist_result, prev

def bellman_ford(g):
    INF = math.inf
    n = g.get_number_of_nodes()
    dist_result = [[INF for _ in range(n)] for _ in range(n)]
    prev = [[None for _ in range(n)] for _ in range(n)]

    for i in range(n):
        dist_result[i][i] = 0

    for source in range(n):
        distance = [INF] * n
        distance[source] = 0

        for _ in range(n - 1):
            for node in range(n):
                for neighbor in g.get_neighbors(node):
                    weight = g.get_weights(node, neighbor)
                    if distance[node] + weight < distance[neighbor]:
                        distance[neighbor] = distance[node] + weight
                        prev[source][neighbor] = node

        dist_result[source] = distance

    return dist_result, prev


    
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

# dist, prev = dijkstra(g)
dist, prev = bellman_ford(g)
print("Distance for all pairs: \n")
for row in dist:
    print(row)
# print(dist[5][0])
# print(prev[5][0])

print("\n \n")

print("Second-to-last vertex:")
for row in prev:
    print(row)