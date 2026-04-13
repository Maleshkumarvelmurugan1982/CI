size = int(input("Grid size: "))

def get_pos(msg):
    print(msg)
    r = int(input("Row: "))
    c = int(input("Col: "))
    return [r, c]

player = get_pos("Your start")
wumpus = get_pos("Wumpus")
gold   = get_pos("Gold")
n = int(input("How many pits? "))
pits = []
for i in range(n):
    print("Pit", i+1)
    p = get_pos("")
    pits.append(p)

def show():
    print()
    for r in range(size):
        for c in range(size):
            if [r, c] == player:   print("K", end=" ")
            elif [r, c] == wumpus: print("W", end=" ")
            elif [r, c] == gold:   print("G", end=" ")
            elif [r, c] in pits:   print("P", end=" ")
            else:                  print("-", end=" ")
        print()

def nearby():
    r, c = player
    return [[r-1,c],[r+1,c],[r,c-1],[r,c+1]]

def sensors():
    adj = nearby()
    print("Smell  :", wumpus in adj)
    print("Breeze :", any(p in adj for p in pits))
    print("Glitter:", player == gold or gold in adj)

def move(cmd):
    r, c = player
    if cmd == "up":    r -= 1
    if cmd == "down":  r += 1
    if cmd == "left":  c -= 1
    if cmd == "right": c += 1
    if 0 <= r < size and 0 <= c < size:
        player[0], player[1] = r, c
    else:
        print("Bump! Wall.")

show()
sensors()

while True:
    cmd = input("Move (up/down/left/right/grab): ").lower()
    if cmd == "grab":
        if player == gold:
            print("You got the gold! YOU WIN!")
        else:
            print("No gold here.")
        break
    move(cmd)
    show()
    sensors()
    if player == wumpus:
        print("Wumpus ate you! GAME OVER.")
        break
    if player in pits:
        print("You fell in a pit! GAME OVER.")
        break
