# k-NN Classifier (n Features, Random Training Data)
# Displays CSV + Entity Count + Rank + Steps
# Prints Min & Max used in Normalization (4-decimal)
# Prints Original and Normalized Training Data in a table (4-decimal precision)
# For each unknown data point prints a table:
#   Original | Normalized | Pred(k1) | Pred(k2) | ...
# Supports weighted and unweighted voting
# Prints Features, Distance, Rank for neighbors (4-decimal)
# No external libraries
# File path set to: C:\Users\males\Downloads\diabetes.csv

import copy

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


# -------- Min-Max Normalization (n features) --------
def normalize(data):
    if not data:
        return [], [], []

    n_features = len(data[0][0])
    mins = [data[0][0][i] for i in range(n_features)]
    maxs = [data[0][0][i] for i in range(n_features)]

    for d in data:
        for i in range(n_features):
            if d[0][i] < mins[i]:
                mins[i] = d[0][i]
            if d[0][i] > maxs[i]:
                maxs[i] = d[0][i]

    norm_data = []
    for d in data:
        norm_feat = []
        for i in range(n_features):
            denom = (maxs[i] - mins[i])
            if denom == 0:
                norm_value = 0.0
            else:
                norm_value = (d[0][i] - mins[i]) / denom
            norm_feat.append(norm_value)
        norm_data.append((norm_feat, d[1]))

    return norm_data, mins, maxs


def normalize_query(q, mins, maxs):
    norm = []
    for i in range(len(q)):
        denom = (maxs[i] - mins[i])
        if denom == 0:
            norm.append(0.0)
        else:
            norm.append((q[i] - mins[i]) / denom)
    return norm


# -------- Distance Metrics (n features) --------
def distance_metrics(p1, p2, metric):
    if metric == "euclidean":
        s = 0.0
        for i in range(len(p1)):
            s += (p1[i] - p2[i]) ** 2
        return s ** 0.5
    elif metric == "manhattan":
        s = 0.0
        for i in range(len(p1)):
            s += abs(p1[i] - p2[i])
        return s
    else:
        raise ValueError("Unsupported metric: choose 'euclidean' or 'manhattan'")


# -------- k-NN with Rank Display (supports weighted/unweighted) --------
def knn_classifier(train, query, k, metric, weighted=False, precision=4):
    distances = []

    # compute distances
    for i, point in enumerate(train):
        d = distance_metrics(point[0], query, metric)
        distances.append((d, point[1], point[0], i))  # distance, label, features, original index

    # sort by distance ascending
    distances.sort(key=lambda x: x[0])

    # print ranked distances table: Rank | Features | Distance | Class
    print("\n--- Ranked Neighbors (Features, Distance, Rank) ---")
    col_rank_w = 6
    col_feat_w = max(30, 12 * len(query))
    col_dist_w = 12
    col_cls_w = 8
    header_row = f"{'Rank':<{col_rank_w}} | {'Features (normalized)':<{col_feat_w}} | {'Distance':<{col_dist_w}} | {'Class':<{col_cls_w}}"
    print(header_row)
    print("-" * len(header_row))
    for rank, item in enumerate(distances, start=1):
        d, label, feats, orig_idx = item
        feat_str = "[" + ", ".join(f"{v:.{precision}f}" for v in feats) + "]"
        d_str = f"{d:.{precision}f}"
        print(f"{rank:<{col_rank_w}} | {feat_str:<{col_feat_w}} | {d_str:<{col_dist_w}} | {label:<{col_cls_w}}")

    # If exact match, return immediately (also print)
    if distances and abs(distances[0][0]) < 1e-15:
        print("\nExact match found (distance 0). Returning class of the exact neighbor.")
        print(f"Rank 1 -> Features: {['{:.4f}'.format(x) for x in distances[0][2]]}, Class: {distances[0][1]}")
        return distances[0][1]

    votes = {}
    eps = 1e-12
    detail_votes = []

    print(f"\n--- {k} Nearest Neighbors (with {'weighted' if weighted else 'unweighted'} votes) ---")
    for i in range(min(k, len(distances))):
        d, label, feats, orig_idx = distances[i]
        if weighted:
            weight = 1.0 / (d + eps)
        else:
            weight = 1.0
        votes[label] = votes.get(label, 0.0) + weight
        feat_str = "[" + ", ".join(f"{v:.{precision}f}" for v in feats) + "]"
        d_str = f"{d:.{precision}f}"
        print(f"Neighbor {i+1} (Rank {i+1}): Features = {feat_str}, Distance = {d_str}, Class = {label}, Weight = {weight:.6f}")
        detail_votes.append((i + 1, d, label, weight))

    print("\n--- Voting Summary ---")
    for cls in sorted(votes.keys()):
        print(f"Class {cls}: {votes[cls]:.6f} (sum of weights)")

    # determine winner (tie-breaker: smallest average distance among tied)
    max_vote = max(votes.values())
    tied = [cls for cls, v in votes.items() if abs(v - max_vote) < 1e-12]

    if len(tied) == 1:
        winner = tied[0]
    else:
        avg_dist = {}
        for cls in tied:
            ds = [d for (_, d, l, _) in detail_votes if l == cls]
            avg_dist[cls] = sum(ds) / len(ds) if ds else float('inf')
        winner = min(avg_dist.items(), key=lambda x: x[1])[0]
        print("\nTie detected. Tie-breaker (smallest average distance among tied classes):")
        for cls in tied:
            print(f"Class {cls}: avg distance = {avg_dist[cls]:.6f}")

    return winner


# -------- Utility: format feature lists to 4-decimal strings --------
def fmt_features(vals, precision=4):
    return "[" + ", ".join(f"{v:.{precision}f}" for v in vals) + "]"


# -------- Main Program --------
if __name__ == "__main__":
    # Use the requested file path
    file_path = r"C:\Users\males\Downloads\diabetes.csv"
    dataset, header = load_csv(file_path)

    print("\nAvailable Features:")
    for i in range(len(header) - 1):
        print(f"{i} -> {header[i]}")

    # Ask user to input feature indices (allow n features)
    features_input = input(
        "\nSelect feature indices (comma-separated) or enter count then indices (e.g., '0,2,4'): "
    ).strip()

    # parse feature indices
    if "," in features_input:
        try:
            selected_indices = [int(x.strip()) for x in features_input.split(",") if x.strip() != ""]
        except ValueError:
            raise ValueError("Invalid feature indices input.")
    else:
        try:
            single = int(features_input)
            more = input(f"You entered '{single}'. Enter 'c' to treat as count of features, or 'i' to treat as a single index: ").strip().lower()
            if more == 'c':
                count = single
                idxs = input(f"Enter {count} feature indices separated by commas: ")
                selected_indices = [int(x.strip()) for x in idxs.split(",") if x.strip() != ""]
                if len(selected_indices) != count:
                    raise ValueError("Number of indices provided does not match the count.")
            else:
                selected_indices = [single]
        except ValueError:
            raise ValueError("Invalid input for features.")

    print(f"\nSelected feature indices: {selected_indices}")
    print("Selected feature names:", [header[i] for i in selected_indices])

    # filter dataset to selected features
    filtered = [([d[0][i] for i in selected_indices], d[1]) for d in dataset]

    train_count = int(input("\nEnter number of training data points (e.g., 15): "))
    training = random_selection(filtered, train_count)

    # keep a copy of original (pre-normalized) selected training data for table
    original_training = [(list(t[0]), t[1]) for t in training]

    print("\n--- Randomly Selected Training Data (Original) ---")
    for i, t in enumerate(original_training, start=1):
        print(f"Point {i}: Features = {t[0]}, Class = {t[1]}")

    # -------- NORMALIZATION --------
    normalized_training, mins, maxs = normalize(training)

    # Print normalization details with 4-decimal precision
    print("\n--- Normalization Details ---")
    for idx, feature_idx in enumerate(selected_indices):
        print(f"Feature {idx} -> {header[feature_idx]}: Min = {mins[idx]:.4f}, Max = {maxs[idx]:.4f}")

    # Print table: Index | Original | Normalized | Class
    print("\n--- Training Data: Original vs Normalized (4-decimal) ---")
    col1_w = 6
    col2_w = max(20, 10 * len(selected_indices))
    col3_w = col2_w
    col4_w = 8
    header_row = f"{'Idx':<{col1_w}} | {'Original':<{col2_w}} | {'Normalized':<{col3_w}} | {'Class':<{col4_w}}"
    print(header_row)
    print("-" * len(header_row))
    for i, (orig, cls) in enumerate(original_training, start=1):
        norm = normalized_training[i-1][0]
        orig_str = fmt_features(orig, precision=4)
        norm_str = fmt_features(norm, precision=4)
        print(f"{i:<{col1_w}} | {orig_str:<{col2_w}} | {norm_str:<{col3_w}} | {cls:<{col4_w}}")

    # Ask for metric and weighting once
    metric = input("\nEnter distance metric (euclidean/manhattan): ").strip().lower()
    weighted_input = input("Use weighted voting? (y/n): ").strip().lower()
    weighted = weighted_input in ("y", "yes", "1", "true")

    # Ask for k values (comma-separated)
    k_input = input("Enter k values (comma-separated, e.g., 3,5,7): ").strip()
    try:
        k_values = [int(x.strip()) for x in k_input.split(",") if x.strip() != ""]
        if not k_values:
            raise ValueError
    except ValueError:
        raise ValueError("Invalid k values input.")

    unknown_count = int(input("\nEnter number of unknown data points: "))

    for u in range(unknown_count):
        print(f"\nEnter unknown data point {u+1}:")
        q = []
        for idx, feature_idx in enumerate(selected_indices):
            val = float(input(f"{header[feature_idx]}: "))
            q.append(val)

        query_norm = normalize_query(q, mins, maxs)

        # compute predictions for each k and print detailed neighbor information inside knn_classifier
        preds = []
        for k in k_values:
            print(f"\n***** Running k-NN for k={k} *****")
            pred = knn_classifier(normalized_training, query_norm, k, metric, weighted, precision=4)
            preds.append(pred)

        # Print table for this unknown: Original | Normalized | Pred(k1) | Pred(k2) | ...
        print("\n--- Prediction Table ---")
        # construct header dynamically
        orig_col = "Original"
        norm_col = "Normalized"
        k_cols = [f"Pred(k={k})" for k in k_values]
        # widths
        col1_w = max(12, 10 * len(selected_indices))
        col2_w = col1_w
        k_w = 20
        header_row = f"{orig_col:<{col1_w}} | {norm_col:<{col2_w}}"
        for kc in k_cols:
            header_row += f" | {kc:<{k_w}}"
        print(header_row)
        print("-" * len(header_row))

        orig_str = fmt_features(q, precision=4)
        norm_str = fmt_features(query_norm, precision=4)
        row = f"{orig_str:<{col1_w}} | {norm_str:<{col2_w}}"
        for pred in preds:
            label_text = f"{pred} ({'Diabetic' if pred == 1 else 'Not Diabetic'})"
            row += f" | {label_text:<{k_w}}"
        print(row)

        # also print a small summary
        print("\nSummary:")
        for k, pred in zip(k_values, preds):
            print(f"  k={k} -> Predicted Class: {pred} ({'Diabetic' if pred == 1 else 'Not Diabetic'})")
