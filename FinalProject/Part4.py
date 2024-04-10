import timeit
from Part1 import MinHeap, Item
from Part3 import A_star, Node
import matplotlib.pyplot as plt
import numpy as np
import csv

class Node:
    def __init__(self, node_id, latitude, longitude, name, display_name, zone, total_lines, rail):
        self.id = node_id
        self.point = (latitude, longitude)
        self.name = name
        self.display_name = display_name
        self.zone = zone
        self.total_lines = total_lines
        self.rail = rail
        self.neighbors = {}

class WeightedGraph2:
    def __init__(self):
        self.nodes = {}
        self.weights = {}

    def add_edge(self, node1, node2, weight):
        node1.neighbors[node2] = weight
        self.weights[(node1.id, node2.id)] = weight

    def add_node(self, node):
        self.nodes[node.id] = node

    def get_weights(self, node1_id, node2_id):  
        # print("Looking up key:", (node1_id, node2_id))  # Debugging line
        return self.weights[(node1_id, node2_id)] 

    def are_connected(self, node1_id, node2_id):
        node1 = self.nodes.get(node1_id)
        if node1:
            return node2_id in node1.neighbors
        return False

    def get_neighbors(self, node_id):
        node = self.nodes.get(node_id)
        if node:
            return list(node.neighbors.keys()) 
        return []

    def get_number_of_nodes(self):
        return len(self.nodes)

    def get_nodes(self):
        return list(self.nodes.keys())


graph = WeightedGraph2()
with open('FinalProject/london_stations.csv', 'r') as station:
    reader = csv.DictReader(station)
    for row in reader:
        node_id = int(row['id'])
        latitude = float(row['latitude'])
        longitude = float(row['longitude'])
        name = row['name']
        display_name = row['display_name']
        zone = float(row['zone'])  # Convert 'zone' to float
        total_lines = int(row['total_lines'])
        rail = int(row['rail'])

        # Create a node and add it to the graph
        node = Node(node_id, latitude, longitude, name, display_name, zone, total_lines, rail)
        graph.add_node(node)

for _, node in graph.nodes.items():
    print(node.id, node.point, node.name, node.display_name, node.zone, node.total_lines, node.rail)




import math  


def calc_weight(p1,p2):
    ax, ay = p1
    bx, by = p2

    weight = math.sqrt((ax-bx) ** 2 + (ay-by)** 2) 
    
    return weight

with open('FinalProject/london_connections.csv', 'r') as file:
    next(file)
    
    for f in file:
        station1, station2, _, _ = f.strip().split(',')
        station1 = int(station1)
        station2 = int(station2)

        node1 = graph.nodes.get(station1)
        node2 = graph.nodes.get(station2)

        weight = calc_weight(node1.point,node2.point)

        graph.add_edge(node1, node2, weight)
        graph.add_edge(node2, node1, weight)

print("Node1","Node2","Weight")
for (n1, n2), weight in graph.weights.items():
    print(n1, n2, weight)




def dijkstra2(G, source, destination):
    dist = {} 
    Q = MinHeap([])
    nodes = list(G.nodes.keys())
    relaxCount = {}
    visited = {}
    prev_line = None

    for node_id in nodes:
        Q.insert(Item(node_id, float("inf"))) 
        dist[node_id] = float("inf")
        relaxCount[node_id] = {'distance': 0, 'transfers': 0}
        visited[node_id] = False

    # Update distance for the source node
    dist[source.id] = 0
    Q.decrease_key(source.id, 0)
    visited[source.id] = True

    while not Q.is_empty() and not visited[destination.id]:
        current_element = Q.extract_min()
        current_node_id = current_element.value  
        dist[current_node_id] = current_element.key
        visited[current_node_id] = True
        
        for neighbor in G.get_neighbors(current_node_id):
            neighbor_id = neighbor.id
            weight = G.get_weights(current_node_id, neighbor_id)

            if not visited[neighbor_id]:  
                temp = dist[current_node_id] + weight
                if temp < dist[neighbor_id]:
                    Q.decrease_key(neighbor_id, temp)
                    dist[neighbor_id] = temp

                    # Update transfer count if line transfer occurs
                    if prev_line is not None and neighbor.total_lines != prev_line:
                        relaxCount[neighbor_id]['transfers'] += 1
                    # Update distance and line information
                    relaxCount[neighbor_id]['distance'] = temp
                    prev_line = neighbor.total_lines
    # print(dist, relaxCount)
    return dist[destination.id], relaxCount[destination.id]


def heuristic(destination, current):
    dx = destination.point[0] - current.point[0]
    dy = destination.point[1] - current.point[1]
    return math.sqrt(dx * dx + dy * dy) 

print(dijkstra2(graph,graph.nodes[11],graph.nodes[30]))

def dictextract(dictarr, index):
    res = []
    for dictionary in dictarr:
        res.append(dictionary[index])
    return res

def experimentplot1(results, title):
    plt.plot(range(0, len(results)), dictextract(results, "astar_time"), label='A*')
    plt.plot(range(0, len(results)), dictextract(results, "dijkstra_time"), label='Dijkstra')
    plt.xlabel('Path #')
    plt.ylabel('Execution time')
    plt.title(title)
    plt.legend()
    plt.show()

def run_experiment(graph):
    results = [] 
    sameline = [] 
    adjline = []
    sevline = []

    # comment out later
    debuglimit = 1
    debugcount = 0

    for source_node_id in graph.get_nodes():  # Iterate over node IDs
        # debug only
        if debugcount >= debuglimit:
            break
        debugcount += 1

        for destination_node_id in graph.get_nodes(): 
            if source_node_id != destination_node_id:



                source_node = graph.nodes.get(source_node_id)  # Get Node object
                destination_node = graph.nodes.get(destination_node_id)  # Get Node object

                start_time = timeit.default_timer()
                dijkstra_distance, relaxCount = dijkstra2(graph, source_node, destination_node)  
                dijkstra_time = timeit.default_timer() - start_time

                start_time = timeit.default_timer()
                A_star(graph, source_node, destination_node, heuristic)
                astar_time = timeit.default_timer() - start_time

                
                temp = {
                    'source': source_node.id,
                    'destination': destination_node.id,
                    'dijkstra_time': dijkstra_time,
                    'astar_time': astar_time,
                    'relax_count': relaxCount["distance"],
                    'transfer_count': relaxCount["transfers"],
                    'dijkstra_distance': dijkstra_distance  
                }
                # print(temp)
                results.append(temp)

                tcount = relaxCount["transfers"]
                if (tcount == 0):
                    sameline.append(temp)
                elif (tcount == 1):
                    adjline.append(temp)
                else:
                    sevline.append(temp)

                transferdict = {}
                try:
                    transferdict[tcount] += 1
                except:
                    transferdict[tcount] = 1
    
    experimentplot1(results, title='Dijkstra vs A* time')

    experimentplot1(sameline, title='Dijkstra vs A* time (same line)')

    experimentplot1(adjline, title='Dijkstra vs A* time (adjacent line)')

    experimentplot1(sevline, title='Dijkstra vs A* time (several line transfers)')

        
    return results


run_experiment(graph)