
# Generate all outcomes as tuples
outcomes = []

for i in range(1, 7):
    for j in range(1, 7):
        outcomes.append((i, j))

total = len(outcomes)   # 36

# Show all outcomes once
print("\nAll Outcomes:")
print(outcomes)


# LOOP
while True:
    query = input("\nEnter query (even / odd / multiple x / less x) or 'exit': ").lower()

    if query == "exit":
        print("Exiting...")
        break

    count = 0
    selected = []

    print("\nMatching Outcomes:")

    # EVEN
    if query == "even":
        for (i, j) in outcomes:
            if (i + j) % 2 == 0:
                print((i, j))
                selected.append((i, j))
                count += 1

    # ODD
    elif query == "odd":
        for (i, j) in outcomes:
            if (i + j) % 2 != 0:
                print((i, j))
                selected.append((i, j))
                count += 1

    # MULTIPLE OF X
    elif "multiple" in query:
        x = int(query.split()[-1])
        for (i, j) in outcomes:
            if (i + j) % x == 0:
                print((i, j))
                selected.append((i, j))
                count += 1

    # LESS THAN X
    elif "less" in query:
        x = int(query.split()[-1])
        for (i, j) in outcomes:
            if (i + j) < x:
                print((i, j))
                selected.append((i, j))
                count += 1

    else:
        print("Invalid input")
        continue


    # Final calculation
    print("\nCalculation:")
    print("Favorable =", count)
    print("Total =", total)
    print("Probability =", count, "/", total, "=", count / total)

    print("\n============================")


sivajothi@sivajothi-HP-Laptop-15s-fq5xxx:~/Desktop$ 
