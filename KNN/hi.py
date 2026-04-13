import csv
import random
import math

def distance_metric(a, b, r):
    return sum(abs(x - y) ** r for x, y in zip(a, b)) ** (1 / r)

def load_data(filename):
    data = []
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if not row or len(row) < 9: 
                    continue
                features = list(map(float, row[:8]))
                label = row[8].strip()
                data.append((features, label))
        return data
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return []

def normalize(train_list, test_point):
    all_features = [item[0] for item in train_list] + [test_point[0]]
    feature_count = len(all_features[0])
    mins = [min(f[i] for f in all_features) for i in range(feature_count)]
    maxs = [max(f[i] for f in all_features) for i in range(feature_count)]

    def scale(f):
        return [(f[i] - mins[i]) / (maxs[i] - mins[i]) if maxs[i] != mins[i] else 0.0 for i in range(feature_count)]

    norm_train = [(scale(f), l, f) for f, l in train_list]
    norm_test = (scale(test_point[0]), test_point[1], test_point[0])
    return norm_train, norm_test

def display_real_data(train_pool, title="REAL DATA FROM DIABETES DATASET"):
    print(f"\n--- {title} ---")
    header = f"{'ID':<4} | {'Preg':<6} | {'Gluc':<6} | {'BP':<6} | {'Skin':<6} | {'Insu':<6} | {'BMI':<7} | {'DPF':<7} | {'Age':<4} | {'Class'}"
    print(header)
    print("-" * len(header))
    for i, (features, label) in enumerate(train_pool, 1):
        preg, gluc, bp, skin, insu, bmi, dpf, age = features
        print(f"{i:<4} | {preg:<6.0f} | {gluc:<6.0f} | {bp:<6.0f} | {skin:<6.0f} | {insu:<6.0f} | {bmi:<7.1f} | {dpf:<7.3f} | {age:<4.0f} | {label}")
    print(f"\nTotal Records Displayed: {len(train_pool)}")

def display_normalization_info(train_pool, test_point):
    all_features = [item[0] for item in train_pool] + [test_point[0]]
    feature_count = len(all_features[0])
    feature_names = get_feature_names()
    
    mins = [min(f[i] for f in all_features) for i in range(feature_count)]
    maxs = [max(f[i] for f in all_features) for i in range(feature_count)]
    
    print("\n" + "="*100)
    print("NORMALIZATION INFORMATION")
    print("="*100)
    print("\nFormula: Normalized_Value = (Original_Value - Min) / (Max - Min)")
    print("\nRange Summary for Each Feature:")
    print("-"*100)
    header = f"{'Feature':<25} | {'Min':<12} | {'Max':<12} | {'Range':<12}"
    print(header)
    print("-"*100)
    
    for i, name in enumerate(feature_names):
        range_val = maxs[i] - mins[i]
        print(f"{name:<25} | {mins[i]:<12.2f} | {maxs[i]:<12.2f} | {range_val:<12.2f}")
    
    print("-"*100)

def display_sample_normalization(train_pool, test_point):
    all_features = [item[0] for item in train_pool] + [test_point[0]]
    feature_count = len(all_features[0])
    feature_names = get_feature_names()
    
    mins = [min(f[i] for f in all_features) for i in range(feature_count)]
    maxs = [max(f[i] for f in all_features) for i in range(feature_count)]
    
    def scale(f):
        return [(f[i] - mins[i]) / (maxs[i] - mins[i]) if maxs[i] != mins[i] else 0.0 for i in range(feature_count)]
    
    normalized_test = scale(test_point[0])
    
    print("\n" + "="*100)
    print("NORMALIZATION - TEST POINT")
    print("="*100)
    print("\nBefore Normalization (Original Values):")
    print("-"*100)
    for i, name in enumerate(feature_names):
        print(f"  {name:<25}: {test_point[0][i]:<10.2f}")
    
    print("\n\nNormalization Process (Step-by-Step):")
    print("-"*100)
    for i, name in enumerate(feature_names):
        orig = test_point[0][i]
        min_val = mins[i]
        max_val = maxs[i]
        range_val = max_val - min_val
        normalized = normalized_test[i]
        
        if range_val != 0:
            formula = f"({orig:.2f} - {min_val:.2f}) / ({max_val:.2f} - {min_val:.2f})"
        else:
            formula = f"Range is 0, so normalized = 0"
        
        print(f"  {name:<25}: {formula:<50} = {normalized:.4f}")
    
    print("\n\nAfter Normalization:")
    print("-"*100)
    for i, name in enumerate(feature_names):
        print(f"  {name:<25}: {normalized_test[i]:<10.4f}")
    print("-"*100)

def display_formatted_table(neighbor_list, title):
    print(f"\n{title}")
    header = f"{'Rank':<6} | {'Features (8 values)':<50} | {'Distance':<15} | {'Class':<8}"
    print(header)
    print("-" * len(header))
    for i, (norm, label, orig, d) in enumerate(neighbor_list, 1):
        feat_str = ", ".join([f"{x:.2f}" for x in orig])
        print(f"{i:<6} | {feat_str:<50} | {d:<15.6f} | {label:<8}")

def display_normalized_neighbors(neighbor_list, title):
    print(f"\n{title} (NORMALIZED VALUES)")
    header = f"{'Rank':<6} | {'Normalized Features':<50} | {'Distance':<15} | {'Class':<8}"
    print(header)
    print("-" * len(header))
    for i, (norm, label, orig, d) in enumerate(neighbor_list, 1):
        feat_str = ", ".join([f"{x:.4f}" for x in norm])
        print(f"{i:<6} | {feat_str:<50} | {d:<15.6f} | {label:<8}")

def get_feature_names():
    return ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

def get_valid_k_value(max_k):
    while True:
        try:
            k = int(input(f"Enter K value for classification (Max {max_k}, must be ODD): "))
            
            if k <= 0:
                print(f"ERROR: K must be a positive number. You entered: {k}")
                print(f"Please try again.\n")
                continue
            
            if k > max_k:
                print(f" ERROR: K cannot be greater than {max_k} (training pool size). You entered: {k}")
                print(f"Please try again.\n")
                continue
            
            if k % 2 == 0:
                print(f"\n{'*'*80}")
                print(f" ERROR: K VALUE MUST BE ODD!")
                print(f"{'*'*80}")
                print(f"You entered K = {k}, which is EVEN.")
                print(f"\nWhy? KNN with even K can result in TIES:")
                print(f"  Example: K=4 neighbors -> 2 vote for Class 0, 2 vote for Class 1")
                print(f"  Result: No clear winner! (TIE)")
                print(f"\nSolution: Use ODD numbers like 1, 3, 5, 7, 9, 11, 13, 15, ...")
                print(f"{'*'*80}\n")
                continue
            
            print(f"K value accepted: {k} (ODD number)")
            return k
            
        except ValueError:
            print(f" ERROR: Invalid input. K must be an integer number.")
            print(f"Please try again.\n")
            continue

def get_number_of_test_points(max_available):
    while True:
        try:
            num_tests = int(input(f"Enter number of test points for evaluation (Max {max_available}): "))
            
            if num_tests <= 0:
                print(f"ERROR: Number of test points must be positive. You entered: {num_tests}")
                print(f"Please try again.\n")
                continue
            
            if num_tests > max_available:
                print(f"ERROR: Number of test points cannot exceed {max_available}. You entered: {num_tests}")
                print(f"Please try again.\n")
                continue
            
            print(f"Number of test points accepted: {num_tests}")
            return num_tests
            
        except ValueError:
            print(f"ERROR: Invalid input. Must be an integer number.")
            print(f"Please try again.\n")
            continue

def classify_single_point(test_point, train_pool, r_val, k, feature_names):
    """Classify a single test point and return predictions"""
    norm_train, norm_test = normalize(train_pool, test_point)
    test_norm, test_label, test_orig = norm_test
    
    results = []
    for train_norm, train_label, train_orig in norm_train:
        dist = distance_metric(test_norm, train_norm, r_val)
        results.append((train_norm, train_label, train_orig, dist))
    
    results.sort(key=lambda x: x[3])
    neighbors = results[:k]
    
    unweighted_counts = {}
    weighted_counts = {}
    for _, label, _, dist in neighbors:
        unweighted_counts[label] = unweighted_counts.get(label, 0) + 1
        # Handle zero distance: if distance is 0, use a very high weight
        if dist == 0:
            weight = 1e10  # Very high weight for exact match
        else:
            weight = 1 / (dist * dist)
        weighted_counts[label] = weighted_counts.get(label, 0) + weight
    
    pred_unweighted = max(unweighted_counts, key=unweighted_counts.get)
    pred_weighted = max(weighted_counts, key=weighted_counts.get)
    
    return {
        'actual': test_label,
        'pred_unweighted': pred_unweighted,
        'pred_weighted': pred_weighted,
        'unweighted_counts': unweighted_counts,
        'weighted_counts': weighted_counts,
        'neighbors': neighbors,
        'test_orig': test_orig
    }

def main():
    raw_data = load_data("diabetes.csv")
    if not raw_data: 
        return

    print(f"Dataset loaded. Total records: {len(raw_data)}")
    
    random.shuffle(raw_data)
    
    # Keep one main test point for initial demonstration
    main_test_point = raw_data.pop()
    print(f"Removed 1 for main test point. Remaining: {len(raw_data)}")
    
    class0 = [item for item in raw_data if item[1] == '0']
    class1 = [item for item in raw_data if item[1] == '1']

    print(f"Class 0 (No Diabetes): {len(class0)}, Class 1 (Diabetes): {len(class1)}")
    limit = int(input(f"Enter number of samples per class (Max {min(len(class0), len(class1))}): "))
    
    # CREATE TRAINING POOL - Remove these from raw_data
    train_pool = class0[:limit] + class1[:limit]
    
    # Remove training samples from raw_data to avoid duplication
    raw_data = [item for item in raw_data if item not in train_pool]
    print(f"Training pool created: {len(train_pool)} records ({limit} from each class)")
    print(f"Remaining records after removing training pool: {len(raw_data)}")
    
    random.shuffle(train_pool) 
    
    display_real_data(train_pool, "TRAINING DATA FROM DIABETES DATASET")

    print(f"\n--- SELECTION SUMMARY ---")
    print(f"Total balanced training points taken: {len(train_pool)} ({limit} from Class 0, {limit} from Class 1)")

    norm_train, norm_test = normalize(train_pool, main_test_point)
    test_norm, test_label, test_orig = norm_test

    feature_names = get_feature_names()
    
    display_normalization_info(train_pool, main_test_point)
    display_sample_normalization(train_pool, main_test_point)
    
    print("\n" + "="*80)
    print(f"MAIN TEST POINT (Features): ")
    for i, name in enumerate(feature_names):
        print(f"  {name}: {test_orig[i]:.2f}")
    print(f"ACTUAL CLASS: {'No Diabetes (0)' if test_label == '0' else 'Diabetes (1)'}")
    print("="*80)

    r_val = int(input("\nEnter distance metric r (1: Manhattan, 2: Euclidean): "))
    
    # Get K value
    k = get_valid_k_value(len(train_pool))
    
    # ===== MAIN TEST POINT CLASSIFICATION =====
    print("\n" + "="*80)
    print("CLASSIFYING MAIN TEST POINT")
    print("="*80)
    
    results = []
    for train_norm, train_label, train_orig in norm_train:
        dist = distance_metric(test_norm, train_norm, r_val)
        results.append((train_norm, train_label, train_orig, dist))

    results.sort(key=lambda x: x[3])
    neighbors = results[:k]
    
    display_formatted_table(neighbors, f"--- TOP {k} NEIGHBORS (ORIGINAL VALUES) ---")
    display_normalized_neighbors(neighbors, f"--- TOP {k} NEIGHBORS")

    unweighted_counts, weighted_counts = {}, {}
    for _, label, _, dist in neighbors:
        unweighted_counts[label] = unweighted_counts.get(label, 0) + 1
        # Handle zero distance in main test point
        if dist == 0:
            weight = 1e10
        else:
            weight = 1 / (dist * dist)
        weighted_counts[label] = weighted_counts.get(label, 0) + weight

    pred_unweighted = max(unweighted_counts, key=unweighted_counts.get)
    pred_weighted = max(weighted_counts, key=weighted_counts.get)

    print("\n" + "*"*80)
    print(f"FINAL RESULTS FOR MAIN TEST POINT")
    print(f"ACTUAL CLASS: {'No Diabetes (0)' if test_label == '0' else 'Diabetes (1)'}")
    print("-" * 80)
    print(f"UNWEIGHTED PREDICTION: CLASS {pred_unweighted} ({'No Diabetes' if pred_unweighted == '0' else 'Diabetes'}) | {unweighted_counts}")
    print(f"WEIGHTED PREDICTION:   CLASS {pred_weighted} ({'No Diabetes' if pred_weighted == '0' else 'Diabetes'}) | { {k: round(v, 3) for k, v in weighted_counts.items()} }")
    print("*"*80)
    
    # ===== MULTIPLE TEST POINTS FOR CONFUSION MATRIX =====
    print("\n" + "="*80)
    print("GENERATING CONFUSION MATRIX WITH MULTIPLE TEST POINTS")
    print("="*80)
    
    # Separate remaining data into classes
    remaining_class0 = [item for item in raw_data if item[1] == '0']
    remaining_class1 = [item for item in raw_data if item[1] == '1']
    
    max_test_points = len(remaining_class0) + len(remaining_class1)
    num_test_points = get_number_of_test_points(max_test_points)
    
    # Balance the test set
    test_points_per_class = num_test_points // 2
    test_pool = remaining_class0[:test_points_per_class] + remaining_class1[:test_points_per_class]
    random.shuffle(test_pool)
    
    print(f"\nTesting with {len(test_pool)} test points ({test_points_per_class} from Class 0, {test_points_per_class} from Class 1)")
    
    # ===== DISPLAY TEST DATA TAKEN =====
    print("\n" + "="*100)
    print("TEST DATA TAKEN FOR EVALUATION")
    print("="*100)
    display_real_data(test_pool, "TEST DATA FROM DIABETES DATASET")
    # Classification for confusion matrix
    tp, tn, fp, fn = 0, 0, 0, 0
    
    print("\n" + "="*100)
    print("CLASSIFICATION RESULTS FOR EACH TEST POINT")
    print("="*100)
    
    for idx, test_point in enumerate(test_pool, 1):
        result = classify_single_point(test_point, train_pool, r_val, k, feature_names)
        
        actual = result['actual']
        pred = result['pred_unweighted']
        
        if actual == '1' and pred == '1': 
            tp += 1
        elif actual == '0' and pred == '0': 
            tn += 1
        elif actual == '0' and pred == '1': 
            fp += 1
        elif actual == '1' and pred == '0': 
            fn += 1
        
        match_status = 'CORRECT' if actual == pred else 'INCORRECT'
        print(f"Test {idx:3d}: Actual={actual}, Predicted={pred}, {match_status}")
    
    print("\n" + "="*60)
    print("      CONFUSION MATRIX (2x2)")
    print("="*60)
    print(f"                     Predicted Negative  Predicted Positive")
    print(f"Actual Negative (0)  |    {tn:<7}         {fp:<7}")
    print(f"Actual Positive (1)  |    {fn:<7}         {tp:<7}")
    print("-" * 60)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"ACCURACY:   {accuracy:.4f} ({(tp + tn)}/{(tp + tn + fp + fn)})")
    print(f"PRECISION:  {precision:.4f}")
    print(f"RECALL:     {recall:.4f} ")
    print(f"SPECIFICITY:{specificity:.4f}")
    print(f"F-SCORE:    {f_score:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
