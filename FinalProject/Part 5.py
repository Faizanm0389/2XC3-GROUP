from typing import Dict, List
from abc import ABC, abstractmethod
from Part3 import A_star
import math

class Graph(ABC):
    def __init__(self, nodes):
        self.graph=[]
        self.weights={}
        for node in range(nodes):
            self.graph.append([])

    def get_adj_nodes(self, node: int) -> List[int]:
        return self.graph[node]
    
    def add_node(self, node: int):
        self.graph[node]=[]

    def add_edge(self, start: int, end: int, w: float):
        if start not in self.graph[start]:
            self.graph[start].append(end)
        self.weights[(start, end)] = w

    def get_num_of_nodes(self) -> int:
        return len(self.graph)

    @abstractmethod 
    def w(self, node):
        pass

class WeightedGraph(Graph):

    def w(self, node1:int, node2:int):
        for neighbour in self.graph[node1]:
            if neighbour == node2:
                return self.weights[(node1, node2)]
    

class HeuristicGraph(WeightedGraph):

    def __init__(self, nodes):
        super().__init__(nodes)
        self.heuristic = {}

    def set_heuristic(self, node: int, value: float):
        self.heuristic[node] = value

    def get_heuristic(self) -> Dict[int, float]:
        return self.heuristic

class SPAlgorithm(ABC):

    @abstractmethod
    def calc_sp(self, graph: Graph) -> float:
        pass

class Dijkstra(SPAlgorithm):

    def calc_sp(self, g: Graph):
        INF = math.inf
        n = g.get_num_of_nodes()
        dist_result = [[INF for _ in range(n)] for _ in range(n)]
        prev = [[None for _ in range(n)] for _ in range(n)]

        for source_node in range(n):
            dist = [INF] * n
            dist[source_node] = 0
            marked = [False] * n

            while True:
                min_distance = INF
                min_node = -1
                for node in range(n):
                    if not marked[node] and dist[node] < min_distance:
                        min_distance = dist[node]
                        min_node = node

                if min_node == -1:
                    break

                marked[min_node] = True

                for neighbor in g.get_adj_nodes(min_node):
                    distance = dist[min_node] + g.w(min_node, neighbor)
                    if distance < dist[neighbor]:
                        dist[neighbor] = distance
                        prev[source_node][neighbor] = min_node

            for dst in range(n):
                dist_result[source_node][dst] = dist[dst]

        return dist_result, prev

class Bellman_Ford(SPAlgorithm):

    def calc_sp(self, g: Graph):
        INF = math.inf
        n = g.get_num_of_nodes()
        dist_result = [[INF for _ in range(n)] for _ in range(n)]
        prev = [[None for _ in range(n)] for _ in range(n)]

        for i in range(n):
            dist_result[i][i] = 0

        for source_node in range(n):
            distance = [INF] * n
            distance[source_node] = 0

            for _ in range(n - 1):
                for node in range(n):
                    for neighbor in g.get_adj_nodes(node):
                        weight = g.w(node, neighbor)
                        if distance[node] + weight < distance[neighbor]:
                            distance[neighbor] = distance[node] + weight
                            prev[source_node][neighbor] = node

            dist_result[source_node] = distance

        return dist_result, prev
    
    
class A_Star(SPAlgorithm):
    def __init__(self):
        self.a_star_adapter = AStarAdapter()

    def calc_sp(self, graph: Graph, source: int, dest: int, h) -> float:
        return self.adapter.shortest_path(graph, source, dest,h)

class AStarAdapter:

    def __init__(self):
        self.a_star = A_star()

    def shortest_path(self, graph: Graph, source: int, dest: int, h) -> float:
        return self.a_star.calc_sp(graph, source, dest,h)

class ShortPathFinder:

    def __init__(self):
        self.graph = None
        self.algorithm = None

    def calc_short_path(self) -> float:
        if self.graph is None or self.algorithm is None:
            raise ValueError("Graph or Algorithm cannot be None value")
        return self.algorithm.calc_sp(self.graph)

    def set_graph(self, graph: Graph):
        self.graph = graph

    def set_algorithm(self, algorithm: SPAlgorithm):
        self.algorithm = algorithm