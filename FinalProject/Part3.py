import math

class Item:
    def __init__(self, value, key):
        self.key = key
        self.value = value

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

def A_star(graph, source, destination, heuristic):
    visited = {}
    distance = {}
    predecessors = {}

    Q = MinHeap([])

    for i in range(graph.get_number_of_nodes()):
        visited[i] = False
        distance[i] = float("inf")
        predecessors[i] = None

    distance[source] = 0
    Q.insert(Item(source, heuristic(destination, source)))

    while not Q.is_empty():
        current_item = Q.extract_min()
        current_node = current_item.value

        if current_node == destination:
            path = []
            while current_node is not None:
                path.insert(0, current_node)
                current_node = predecessors[current_node]
            return predecessors, path

        visited[current_node] = True

        for neighbour in graph.graph[current_node]:
            edge_weight = graph.get_weights(current_node, neighbour)

            temp = distance[current_node] + edge_weight

            if not visited[neighbour] and temp < distance[neighbour]:
                distance[neighbour] = temp
                predecessors[neighbour] = current_node
                Q.insert(Item(neighbour, temp + heuristic(destination, neighbour)))

    return {}, []

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
g.add_edge(5,4,15)

def simple_heuristic(destination, node):
    """A very basic heuristic: straight-line distance to the destination (example)."""
    return abs(destination - node) 

predecessors, path = A_star(g, 0, 5, simple_heuristic) 
print("Predecessors:", predecessors)
print("Shortest Path:", path)