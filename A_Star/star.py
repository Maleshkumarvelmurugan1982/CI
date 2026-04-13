import heapq

graph = {}
h = {}

def add_node(n, hv):
    graph[n] = {}
    h[n] = hv

def delete_node(n):
    graph.pop(n, None)
    h.pop(n, None)
    for i in graph:
        graph[i].pop(n, None)

def add_edge(a,b,c):
    graph[a][b] = c
    graph[b][a] = c

def delete_edge(a,b):
    graph[a].pop(b, None)
    graph[b].pop(a, None)

def display():
    print(graph)

def astar(start,goal):
    open = [(0,start)]
    g = {i:999 for i in graph}
    g[start] = 0
    parent = {start:None}

    while open:
        f,node = heapq.heappop(open)

        if node == goal:
            path=[]
            while node:
                path.append(node)
                node=parent[node]
            print("Path:",path[::-1])
            return

        for n,c in graph[node].items():
            new = g[node] + c
            if new < g[n]:
                g[n] = new
                heapq.heappush(open,(new+h[n],n))
                parent[n] = node


while True:
    print("\n1 add node 2 del node 3 add edge 4 del edge 5 show 6 A* 7 exit")
    ch=int(input())

    if ch==1:
        n=input("node:")
        hv=int(input("heuristic:"))
        add_node(n,hv)

    elif ch==2:
        n=input("node:")
        delete_node(n)

    elif ch==3:
        a=input("from:")
        b=input("to:")
        c=int(input("cost:"))
        add_edge(a,b,c)

    elif ch==4:
        a=input("from:")
        b=input("to:")
        delete_edge(a,b)

    elif ch==5:
        display()

    elif ch==6:
        s=input("start:")
        g=input("goal:")
        astar(s,g)

    else:
        break
