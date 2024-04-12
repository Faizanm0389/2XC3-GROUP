import math 
import random
import timeit
import matplotlib.pyplot as plt
import numpy as np


class MinHeap:
    def __init__(self, data):
        self.items = data
        self.length = len(data)
        self.build_heap()

        # add a map based on input node
        self.map = {}
        for i in range(self.length):
            self.map[self.items[i].value] = i

    def find_left_index(self,index):
        return 2 * (index + 1) - 1

    def find_right_index(self,index):
        return 2 * (index + 1)

    def find_parent_index(self,index):
        return (index + 1) // 2 - 1  
    
    def sink_down(self, index):
        smallest_known_index = index

        if self.find_left_index(index) < self.length and self.items[self.find_left_index(index)].key < self.items[index].key:
            smallest_known_index = self.find_left_index(index)

        if self.find_right_index(index) < self.length and self.items[self.find_right_index(index)].key < self.items[smallest_known_index].key:
            smallest_known_index = self.find_right_index(index)

        if smallest_known_index != index:
            self.items[index], self.items[smallest_known_index] = self.items[smallest_known_index], self.items[index]
            
            # update map
            self.map[self.items[index].value] = index
            self.map[self.items[smallest_known_index].value] = smallest_known_index

            # recursive call
            self.sink_down(smallest_known_index)

    def build_heap(self,):
        for i in range(self.length // 2 - 1, -1, -1):
            self.sink_down(i) 

    def insert(self, node):
        if len(self.items) == self.length:
            self.items.append(node)
        else:
            self.items[self.length] = node
        self.map[node.value] = self.length
        self.length += 1
        self.swim_up(self.length - 1)

    def insert_nodes(self, node_list):
        for node in node_list:
            self.insert(node)

    def swim_up(self, index):
        
        while index > 0 and self.items[self.find_parent_index(index)].key < self.items[self.find_parent_index(index)].key:
            #swap values
            self.items[index], self.items[self.find_parent_index(index)] = self.items[self.find_parent_index(index)], self.items[index]
            #update map
            self.map[self.items[index].value] = index
            self.map[self.items[self.find_parent_index(index)].value] = self.find_parent_index(index)
            index = self.find_parent_index(index)

    def get_min(self):
        if len(self.items) > 0:
            return self.items[0]

    def extract_min(self,):
        #xchange
        self.items[0], self.items[self.length - 1] = self.items[self.length - 1], self.items[0]
        #update map
        self.map[self.items[self.length - 1].value] = self.length - 1
        self.map[self.items[0].value] = 0

        min_node = self.items[self.length - 1]
        self.length -= 1
        self.map.pop(min_node.value)
        self.sink_down(0)
        return min_node

    def decrease_key(self, value, new_key):
        if new_key >= self.items[self.map[value]].key:
            return
        index = self.map[value]
        self.items[index].key = new_key
        self.swim_up(index)

    def get_element_from_value(self, value):
        return self.items[self.map[value]]

    def is_empty(self):
        return self.length == 0
    
    def __str__(self):
        height = math.ceil(math.log(self.length + 1, 2))
        whitespace = 2 ** height + height
        s = ""
        for i in range(height):
            for j in range(2 ** i - 1, min(2 ** (i + 1) - 1, self.length)):
                s += " " * whitespace
                s += str(self.items[j]) + " "
            s += "\n"
            whitespace = whitespace // 2
        return s
class Item:
    def __init__(self, value, key):
        self.key = key
        self.value = value
    
    def __str__(self):
        return "(" + str(self.key) + "," + str(self.value) + ")"
    
class WeightedGraph:
    def __init__(self, nodes):
        self.graph = [[] for _ in range(nodes)]
        self.weights = {}

    def add_edge(self, node1, node2, weight):
        self.graph[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def get_weights(self, node1, node2):
        return self.weights.get((node1, node2))

    def get_neighbors(self, node):
        return self.graph[node]

    def get_number_of_nodes(self):
        return len(self.graph)

    def get_nodes(self):
        return [i for i in range(len(self.graph))]
    
    def get_edges(self):
        edges = []
        for node1 in range(len(self.graph)):
            for node2 in self.graph[node1]:
                weight = self.get_weights(node1, node2)
                edges.append((node1, node2, weight))
        return edges

    def __iter__(self):
        return iter(self.graph)



#part 1.1

def dijkstra_relax(Graph, source, k):
    visited = {}
    distance = {}
    relax_count = {}  
    predecessor = {} 

    Q = MinHeap([])

    for i in range(Graph.get_number_of_nodes()):
        visited[i] = False
        distance[i] = float("inf")
        relax_count[i] = 0 
        predecessor[i] = None  

        Q.insert(Item(i, 0))


    Q.decrease_key(source, 0)
    distance[source] = 0

    while not Q.is_empty():
        current_node = Q.extract_min().value
        visited[current_node] = True

        for neighbour in Graph.graph[current_node]:
            edge_weight = Graph.get_weights(current_node, neighbour)
            temp = distance[current_node] + edge_weight
            if not visited[neighbour]:
                if temp < distance[neighbour]:
                    distance[neighbour] = temp
                    Q.decrease_key(neighbour, temp)
                    relax_count[neighbour] += 1  
                    predecessor[neighbour] = current_node  
        if relax_count[current_node] >= k:
            break

    def k_path(node, predecessor, path=[]):
        if node is None:
            return path
        path.insert(0, node)
        return k_path(predecessor[node], predecessor, path)
    shortest_paths = {}
    node = 0
    while True:
        shortest_paths[node] = k_path(node, predecessor)
        node += 1
        if node >= Graph.get_number_of_nodes():
            break

    return distance, shortest_paths



#part 1.2

def bellamford_relax(graph, source, k):
  
    distance = {node: float('inf') for node in graph.get_nodes()}
    predecessor = {node: None for node in graph.get_nodes()}
    distance[source] = 0

    for _ in range(k):
        relaxed = False
        for u, v, weight in graph.get_edges():
            if distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
                predecessor[v] = u
                relaxed = True
        else:
            break

        for u in graph.get_nodes():
            for v in graph.get_neighbors(u):
                weight = graph.get_weights(u, v)
                if distance[u] + weight < distance[v]:
                    return None

    def k_path(node, predecessor, path=[]):
        if node is None:
            return path
        path.insert(0, node)
        return k_path(predecessor[node], predecessor, path)

    shortest_paths = {}
    for node in graph.get_nodes():
        shortest_paths[node] = k_path(node, predecessor)

    return distance, shortest_paths

#part 1.3

def gen_graph(n):
    graph = WeightedGraph(n)
    
    for n1 in range(n):
        for n2 in range(n1 + 1, n):  
            weight = random.randint(1, 10) 
            graph.add_edge(n1, n2, weight)
            graph.add_edge(n2, n1, weight)
    
    return graph

def experiment_graph_size():

  bellman_time = []
  dijkstra_time = []

  for n in range(10, 1000, 50):
    g1 = gen_graph(n)
    g2 = g1  

    # belman 
    start = timeit.default_timer()
    bellamford_relax(g1, 0, n-1)
    stop = timeit.default_timer()
    bellman_time.append(stop - start)

    # dijkst
    start = timeit.default_timer()
    dijkstra_relax(g2, 0, n-1)
    stop = timeit.default_timer()
    dijkstra_time.append(stop - start)


  plt.plot(np.arange(10, 1000, 50), bellman_time, label='Bellman-Ford')
  plt.plot(np.arange(10, 1000, 50), dijkstra_time, label='Dijkstra')
  plt.xlabel('Size N (10-1000)')
  plt.ylabel('Execution time')
  plt.title('Experiment Graph Size ')
  plt.legend()
  plt.show()

def experiment_graph_density():
    bellman_time = []
    dijkstra_time = []
    graph = WeightedGraph(1000)
    for i in range(100):
        graph.add_edge(i, i, 0)
        
    for i in range(1, 1000, 100):
        for j in range(i):
            n1 = random.randint(0, 999)
            n2 = random.randint(0, 999)
            w = random.randint(1, 10)
            graph.add_edge(n1, n2, w)

        start = timeit.default_timer()
        bellamford_relax(graph, 0, 999)
        stop = timeit.default_timer()
        bellman_time.append(stop - start)

        start = timeit.default_timer()
        dijkstra_relax(graph, 0, 999)
        stop = timeit.default_timer()
        dijkstra_time.append(stop - start)

    plt.plot(np.arange(0, 1000, 100), bellman_time, label='Bellman-Ford')
    plt.plot(np.arange(0, 1000, 100), dijkstra_time, label='Dijkstra')
    plt.xlabel('Density')
    plt.ylabel('Execution time')
    plt.title('Experiment Graph Density')
    plt.legend()
    plt.show()

def experiment_graph_relaxation():
    bellman_times = []
    dijkstra_times = []
    g = gen_graph(1000)

    for k in range(1, 1000, 50):
        # bellman
        start = timeit.default_timer()
        bellamford_relax(g, 0, k)
        stop = timeit.default_timer()
        bellman_times.append(stop - start)

        #  Dijkstra
        start = timeit.default_timer()
        dijkstra_relax(g, 0, k)
        stop = timeit.default_timer()
        dijkstra_times.append(stop - start)

    plt.plot(range(1, 1000, 50), bellman_times, label='Bellman-Ford')
    plt.plot(range(1, 1000, 50), dijkstra_times, label='Dijkstra')
    plt.xlabel('Relaxation limit from source node')
    plt.ylabel('Execution time')
    plt.title('Experiment Relaxation limit ')
    plt.legend()
    plt.show()   



print(experiment_graph_size())
print(experiment_graph_relaxation())
print(experiment_graph_density())
