import heapq

class Graph:
    def __init__(self):
        self.graph = {}

    # Add node
    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []
            print("Node added successfully")
        else:
            print("Node already exists")

    # Delete node
    def delete_node(self, node):
        if node in self.graph:
            del self.graph[node]
            for n in self.graph:
                self.graph[n] = [(nbr, wt) for nbr, wt in self.graph[n] if nbr != node]
            print("Node deleted successfully")
        else:
            print("Node not found")

    # Add edge with cost
    def add_edge(self, node1, node2, cost):
        if node1 in self.graph and node2 in self.graph:
            self.graph[node1].append((node2, cost))
            self.graph[node2].append((node1, cost))
            print("Edge added successfully")
        else:
            print("One or both nodes not found")

    # Delete edge
    def delete_edge(self, node1, node2):
        if node1 in self.graph and node2 in self.graph:
            self.graph[node1] = [(nbr, wt) for nbr, wt in self.graph[node1] if nbr != node2]
            self.graph[node2] = [(nbr, wt) for nbr, wt in self.graph[node2] if nbr != node1]
            print("Edge deleted successfully")
        else:
            print("Nodes not found")

    # Display graph
    def display(self):
        if not self.graph:
            print("Graph is empty")
        else:
            for node in self.graph:
                print(node, "->", self.graph[node])

    # BFS
    def bfs(self, start):
        if start not in self.graph:
            print("Start node not found")
            return

        visited = []
        queue = [start]

        visited.append(start)

        print("BFS Traversal:")
        while queue:
            node = queue.pop(0)
            print(node, end=" ")

            for neighbour, _ in self.graph[node]:
                if neighbour not in visited:
                    visited.append(neighbour)
                    queue.append(neighbour)
        print()

    # DFS
    def dfs(self, start):
        if start not in self.graph:
            print("Start node not found")
            return

        visited = []
        stack = [start]

        visited.append(start)

        print("DFS Traversal:")
        while stack:
            node = stack.pop()
            print(node, end=" ")

            for neighbour, _ in self.graph[node]:
                if neighbour not in visited:
                    visited.append(neighbour)
                    stack.append(neighbour)
        print()

    # UCS (Uniform Cost Search)
    def ucs(self, start):
        if start not in self.graph:
            print("Start node not found")
            return

        visited = []
        pq = []

        heapq.heappush(pq, (0, start))

        print("UCS Traversal:")
        while pq:
            cost, node = heapq.heappop(pq)

            if node not in visited:
                visited.append(node)
                print(node, "Cost:", cost)

                for neighbour, weight in self.graph[node]:
                    if neighbour not in visited:
                        heapq.heappush(pq, (cost + weight, neighbour))


# -------- MENU --------
g = Graph()

while True:
    print("\n----- GRAPH MENU -----")
    print("1. Add Node")
    print("2. Delete Node")
    print("3. Add Edge (with cost)")
    print("4. Delete Edge")
    print("5. Display Graph")
    print("6. BFS Traversal")
    print("7. DFS Traversal")
    print("8. UCS Traversal")
    print("9. Exit")

    choice = input("Enter your choice (1-9): ")

    if choice == "1":
        node = input("Enter node name: ")
        g.add_node(node)

    elif choice == "2":
        node = input("Enter node to delete: ")
        g.delete_node(node)

    elif choice == "3":
        node1 = input("Enter first node: ")
        node2 = input("Enter second node: ")
        cost = int(input("Enter cost: "))
        g.add_edge(node1, node2, cost)

    elif choice == "4":
        node1 = input("Enter first node: ")
        node2 = input("Enter second node: ")
        g.delete_edge(node1, node2)

    elif choice == "5":
        g.display()

    elif choice == "6":
        start = input("Enter start node for BFS: ")
        g.bfs(start)

    elif choice == "7":
        start = input("Enter start node for DFS: ")
        g.dfs(start)

    elif choice == "8":
        start = input("Enter start node for UCS: ")
        g.ucs(start)

    elif choice == "9":
        print("Exiting program...")
        break

    else:
        print("Invalid choice! Please enter 1 to 9.")
