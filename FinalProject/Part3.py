import math

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

class Item:
    def __init__(self, value, key, node):
        self.key = key
        self.value = value
        self.node = node

    def __lt__(self, other):  # For use with priority queue
        return self.key < other.key 

    def __str__(self):
        return "(" + str(self.key) + "," + str(self.value) + ")"



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

def A_star(graph, source, destination, heuristic):
    visited = {}
    distance = {}
    predecessors = {}

    Q = MinHeap([])

    for node_id in graph.get_nodes():
        visited[node_id] = False
        distance[node_id] = float("inf")
        predecessors[node_id] = None

    distance[source.id] = 0 
    Q.insert(Item(source.id, heuristic(destination, source), source)) 

    while not Q.is_empty():
        current_item = Q.extract_min()
        current_node_id = current_item.value 

        if current_node_id == destination.id:
            # Reconstruct path (exercise for you)
            return predecessors, []  

        visited[current_node_id] = True

        for neighbor in graph.get_neighbors(current_node_id):
            neighbor_id = neighbor.id
            edge_weight = graph.get_weights(current_node_id, neighbor_id)
            temp = distance[current_node_id] + edge_weight

            if not visited[neighbor_id] and temp < distance[neighbor_id]:
                distance[neighbor_id] = temp
                predecessors[neighbor_id] = current_node_id

                neighbor_node = graph.nodes[neighbor_id]  
                item = Item(neighbor_id, temp + heuristic(destination, neighbor_node), neighbor_node) 
                Q.insert(item)

    return {}, [] 