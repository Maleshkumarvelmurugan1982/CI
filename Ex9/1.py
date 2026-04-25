
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

# Print table
def print_table(table):
    print("\n--- Current Joint Probability Table ---")
    for (b,e,a), prob in table.items():
        print(f"P({b},{e},{a}) = {prob}")
    print("--------------------------------------\n")


# Modify table
def modify_table(table):
    print("\nEnter new values (press Enter to keep old value):\n")
    new_table = {}
    for key in table:
        b, e, a = key
        inp = input(f"P({b},{e},{a}) = ")
        if inp.strip() == "":
            new_table[key] = table[key]
        else:
            new_table[key] = float(inp)
    return new_table


# Function with steps
def get_prob(table, conditions, label):
    total = 0
    values = []

    print(f"\n--- {label} ---")
    print("Conditions:", conditions)

    for (b,e,a), prob in table.items():
        row = (b,e,a)
        if all(x in row for x in conditions):
            print(f"P{row} = {prob}")
            values.append(prob)
            total += prob

    print("\nCalculation:")
    print(" + ".join(map(str, values)), "=", total)

    return total


# ---- MAIN ----

table = default_table

# Show table first
print_table(table)

# Ask user to modify
choice = input("Is this table OK? (y/n): ").lower()

if choice == "n":
    table = modify_table(table)
    print("\nUpdated Table:")
    print_table(table)


# LOOP
while True:
    query = input("Enter query (P(a), P(a,b), P(a|b)) or 'exit': ").lower()

    if query == "exit":
        print("Exiting...")
        break

    query = query.replace("p(","").replace(")","")

    # CONDITIONAL
    if "|" in query:
        left, right = query.split("|")
        left = [x.strip() for x in left.split(",")]
        right = list(set([x.strip() for x in right.split(",")]))

        numerator = get_prob(table, left + right, "Numerator P(X,Y)")
        denominator = get_prob(table, right, "Denominator P(Y)")

        if denominator == 0:
            print("\nInvalid (division by zero)")
        else:
            print(f"\nFinal Calculation: {numerator} / {denominator}")
            print("Answer =", numerator / denominator)

    # JOINT
    else:
        terms = [x.strip() for x in query.split(",")]
        result = get_prob(table, terms, "Joint Probability")
        print("\nAnswer =", result)

    print("\n=====================================\n")

sivajothi@sivajothi-HP-Laptop-15s-fq5xxx:~/Desktop$ 
