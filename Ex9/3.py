# Default joint probability table
default_table = {
    ("b","e","a"): 0.01,
    ("b","~e","a"): 0.08,
    ("b","e","~a"): 0.001,
    ("b","~e","~a"): 0.009,
    ("~b","e","a"): 0.01,
    ("~b","~e","a"): 0.09,
    ("~b","e","~a"): 0.01,
    ("~b","~e","~a"): 0.79
}

# Function to get table from user
def get_table():
    print("\nEnter values (press Enter to keep default):\n")
    table = {}

    for key in default_table:
        b, e, a = key
        inp = input(f"P({b},{e},{a}) = ")

        if inp.strip() == "":
            table[key] = default_table[key]
        else:
            table[key] = float(inp)

    return table


# Print table
def print_table(table):
    print("\n--- Current Joint Probability Table ---")
    for (b,e,a), prob in table.items():
        print(f"P({b},{e},{a}) = {prob}")
    print("--------------------------------------\n")


# Function to compute with only matched rows + total
def joint_prob(table, conditions):
    total = 0

    print("\nCalculation:")
    print(" + ".join([f"P({c})" for c in conditions]), "(conceptually)\n")

    print("Matched rows:")
    values = []

    for (b,e,a), prob in table.items():
        row = (b,e,a)

        if all(x in row for x in conditions):
            print(f"P{row} = {prob}")
            values.append(prob)
            total += prob

    # Show summation clearly
    print("\nTotal calculation:")
    print(" + ".join(map(str, values)), "=", total)

    return total


# ---- MAIN ----

choice = input("Use default table? (y/n): ").lower()

if choice == "y":
    table = default_table
else:
    table = get_table()

print_table(table)

# LOOP
while True:
    query = input("Enter query (P(a), P(a,b)) or 'exit': ").lower()

    if query == "exit":
        break

    query = query.replace("p(","").replace(")","")
    terms = [x.strip() for x in query.split(",")]

    result = joint_prob(table, terms)

    print("\nFinal Answer =", result)
    print("\n==============================\n")


sivajothi@sivajothi-HP-Laptop-15s-fq5xxx:~/Desktop$ 
