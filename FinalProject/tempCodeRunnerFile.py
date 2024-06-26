from typing import Dict,List
from abc import ABC, abstractmethod
from Part2 import WeightedGraph
from Part3 import A_star
import math

class Graph(ABC):

    def __init__(self,nodes):
        self.graph = [[] for _ in range(nodes)]
        self.weights = {}

    def get_adj_nodes(self, node:int) -> List[int]:
        return WeightedGraph.get_neighbors(self,node)

    def add_node(self, node:int):
        self.graph.append(node)

    def add_edge(self, start:int, end:int, w:float):
        return WeightedGraph.add_edge(self,start,end,w)

    def get_num_of_nodes(self) -> int:
        return WeightedGraph.get_number_of_nodes(self)
    
    @abstractmethod 
    def w(self, node):
        pass
class WeightedGraph(Graph):

    def w(self,node1:int, node2:int) -> float:
        return WeightedGraph.weights.get((self,node1, node2))
    
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
    def calc_sp(self, graph:Graph, source: int, dest: int) -> float:
        pass

class Dijkstra(SPAlgorithm):

    # from Part 2
    def calc_sp(g):
        INF = math.inf
        n = g.get_number_of_nodes()
        dist_result = [[INF for _ in range(n)] for _ in range(n)]
        prev = [[None for _ in range(n)] for _ in range(n)]

        for source in range(n):
            dist = [INF] * n
            dist[source] = 0
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

                for neighbor in g.get_neighbors(min_node):
                    distance = dist[min_node] + g.get_weights(min_node, neighbor)
                    if distance < dist[neighbor]:
                        dist[neighbor] = distance
                        prev[source][neighbor] = min_node

            for dst in range(n):
                dist_result[source][dst] = dist[dst]

        return dist_result, prev


class Bellman_Ford(SPAlgorithm):

    # from Part 2
    def calc_sp(g):
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
        
class A_Star(SPAlgorithm):

    def __init__(self):
        super().__init__()
        self.adapter = AStarAdapter()

    def calc_sp(self, graph: Graph, source: int, dest: int) -> float:

        return self.adapter.shortest_distance(graph, source, dest)

class AStarAdapter:

    # From Part 3
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
