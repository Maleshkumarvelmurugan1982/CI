# -------------------------------------------------
# k-NN Classifier (2 Features, Random Training Data)
# Displays CSV + Entity Count + Rank + Steps
# Prints Min & Max used in Normalization
# No external libraries
# -------------------------------------------------

# -------- Load CSV --------
def load_csv(file_path):
    file = open(file_path, "r")
    lines = file.readlines()
    file.close()

    print("\n--- FULL CSV DATASET ---")
    for line in lines:
        print(line.strip())

    total_entities = len(lines) - 1
    print("\n>>> Total number of entities (rows in file):", total_entities)

    header = lines[0].strip().split(",")

    data = []
    for i in range(1, len(lines)):
        row = lines[i].strip().split(",")
        features = [float(x) for x in row[:-1]]
        label = int(row[-1])
        data.append((features, label))

    return data, header


# -------- Random Selection --------
def random_selection(data, count):
    selected = []
    used = []
    seed = len(data) + count

    while len(selected) < count:
        seed = (seed * 41 + 17) % len(data)
        if seed not in used:
            used.append(seed)
            selected.append(data[seed])

    return selected


# -------- Min-Max Normalization --------
def normalize(data):
    mins = [data[0][0][0], data[0][0][1]]
    maxs = [data[0][0][0], data[0][0][1]]

    for d in data:
        for i in range(2):
            if d[0][i] < mins[i]:
                mins[i] = d[0][i]
            if d[0][i] > maxs[i]:
                maxs[i] = d[0][i]

    norm_data = []
    for d in data:
        norm_feat = []
        for i in range(2):
            norm_feat.append((d[0][i] - mins[i]) / (maxs[i] - mins[i]))
        norm_data.append((norm_feat, d[1]))

    return norm_data, mins, maxs


def normalize_query(q, mins, maxs):
    return [(q[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(2)]


# -------- Distance Metrics --------
def distance_metrics(p1, p2, metric):
    if metric == "euclidean":
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5
    elif metric == "manhattan":
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


# -------- k-NN with Rank Display --------
def knn_classifier(train, query, k, metric):
    distances = []

    print("\n--- Distance Calculation ---")
    for i, point in enumerate(train):
        d = distance_metrics(point[0], query, metric)
        distances.append((d, point[1]))
        print(f"Point {i+1}: Distance = {d:.4f}, Class = {point[1]}")

    for i in range(len(distances)):
        for j in range(i + 1, len(distances)):
            if distances[i][0] > distances[j][0]:
                distances[i], distances[j] = distances[j], distances[i]

    print("\n--- Ranked Distances ---")
    for rank, d in enumerate(distances, start=1):
        print(f"Rank {rank}: Distance = {d[0]:.4f}, Class = {d[1]}")

    print(f"\n--- {k} Nearest Neighbors ---")
    votes = {}
    for i in range(k):
        label = distances[i][1]
        votes[label] = votes.get(label, 0) + 1
        print(f"Neighbor {i+1} (Rank {i+1}): Class = {label}")

    print("\n--- Voting ---")
    for cls in votes:
        print(f"Class {cls}: {votes[cls]} votes")

    return max(votes, key=votes.get)


# -------- Main Program --------
file_path = "/home/sivajothi/Downloads/diabetes.csv"
dataset, header = load_csv(file_path)

print("\nAvailable Features:")
for i in range(len(header) - 1):
    print(f"{i} -> {header[i]}")

f1 = int(input("\nSelect first feature index: "))
f2 = int(input("Select second feature index: "))

filtered = [([d[0][f1], d[0][f2]], d[1]) for d in dataset]

train_count = int(input("\nEnter number of training data points (e.g., 15): "))
training = random_selection(filtered, train_count)

print("\n--- Randomly Selected Training Data ---")
for i, t in enumerate(training, start=1):
    print(f"Point {i}: {t}")

# -------- NORMALIZATION --------
training, mins, maxs = normalize(training)

print("\n--- Normalization Details ---")
print(f"Min values used: {header[f1]} = {mins[0]}, {header[f2]} = {mins[1]}")
print(f"Max values used: {header[f1]} = {maxs[0]}, {header[f2]} = {maxs[1]}")

k = int(input("\nEnter value of k: "))
metric = input("Enter distance metric (euclidean/manhattan): ")

unknown_count = int(input("\nEnter number of unknown data points: "))

for u in range(unknown_count):
    print(f"\nEnter unknown data point {u+1}:")
    q1 = float(input(header[f1] + ": "))
    q2 = float(input(header[f2] + ": "))

    query = normalize_query([q1, q2], mins, maxs)

    result = knn_classifier(training, query, k, metric)

    print("\n--- Final Prediction ---")
    print("Predicted Class:", result)
    print("→ Diabetic" if result == 1 else "→ Not Diabetic")
