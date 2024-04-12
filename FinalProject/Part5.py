from abc import ABC, abstractmethod
import Part1
from Part3 import A_star
from typing import Dict,List

class Graph:

    def __init__(self,nodes):
        self.graph = [[] for _ in range(nodes)]
        self.weights = {}

    def get_adj_nodes(self, node:int) -> List[int]:
        return Part1.WeightedGraph.get_neighbors(self,node)

    def add_node(self, node:int):
        self.graph.append(node)

    def add_edge(self, start:int, end:int, w:float):
        return Part1.WeightedGraph.add_edge(self,start,end,w)

    def get_num_of_nodes(self) -> int:
        return Part1.WeightedGraph.get_number_of_nodes(self)
    
    def w(self, node1:int, node2:int)-> float:
        return Part1.WeightedGraph.weights.get((self,node1, node2))

class WeightedGraph(Graph):

    def w(self,node1:int, node2:int) -> float:
        return Part1.WeightedGraph.weights.get((self,node1, node2))
    
class HeuristicGraph(WeightedGraph):
    def __init__(self, nodes):
        super().__init__(nodes)
        self.heuristic = {}

    def get_heuristic(self) -> Dict[int, float]:
        return self.heuristic

    def set_heuristic(self, node: int, value: float):
        self.heuristic[node] = value
    

class SPAlgorithm(ABC):

    @abstractmethod
    def calc_sp(self, graph:Graph, source: int, dest: int) -> float:
        pass

class Dijkstra(SPAlgorithm):

    def calc_sp(self,Graph, source, k):
        visited = {}
        distance = {}
        relax_count = {}  
        predecessor = {} 

        Q = Part1.MinHeap([])

        for i in range(Graph.get_number_of_nodes()):
            visited[i] = False
            distance[i] = float("inf")
            relax_count[i] = 0 
            predecessor[i] = None  

            Q.insert(Part1.Item(i, 0))


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


class Bellman_Ford(SPAlgorithm):

    def calc_sp(self,graph, source, k):
    
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
        
class A_Star(SPAlgorithm):

    def __init__(self):
        super().__init__()
        self.adapter = AStarAdapter()

    def calc_sp(self, graph: Graph, source: int, dest: int) -> float:

        return self.adapter.shortest_distance(graph, source, dest)

# Define the adapter class
class AStarAdapter:

    def __init__(self):
        self.a_star = A_star()

    def shortest_distance(self, graph: Graph, source: int, dest: int) -> float:

        return self.a_star(graph, source, dest)


class ShortPathFinder:

    def __init__(self):
        self.Graph = Graph()
        self.SPAlgorithm = SPAlgorithm()

    def calc_short_path(self, source:int, dest:int) -> float:
        return self.SPAlgorithm.calc_sp(self.Graph, source, dest)

    def set_graph(self, graph:Graph):
        self._Graph = graph

    def set_algorithm(self, algorithm:SPAlgorithm):
        self._SPAlgorithm = algorithm
