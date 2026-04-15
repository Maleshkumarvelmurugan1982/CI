import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

dataset_name = input("Enter dataset CSV filename: ")
n_trees = int(input("Enter number of decision trees: "))
splits_input = input("Enter splits: ")

splits = []
split_labels = []

for s in splits_input.split(","):
    train_p, test_p = map(int, s.split("-"))
    if train_p + test_p != 100:
        print(f"Invalid Split {s}, must sum to 100, skipping.")
        continue
    test_size = test_p / 100
    splits.append(test_size)
    split_labels.append(s)

data = pd.read_csv(dataset_name)
print("\nColumns in dataset:", list(data.columns))

target_col = input("Enter target column name: ")
x = data.drop(columns=[target_col])
y = data[target_col]

results = []

for i in range(len(splits)):
    split = splits[i]
    label = split_labels[i]
    print(f"\nProcessing Split: {label}")
   
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split, random_state=42)
   
    # Fixed: RandomForestClassifier (was RandomForectClassifier)
    model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    model.fit(x_train, y_train)
   
    y_pred = model.predict(x_test)
   
    # Fixed: accuracy_score (was accracy_score)
    acc = accuracy_score(y_test, y_pred)
    # Fixed: zero_division (was zero_divison)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
   
    results.append([label, acc, prec, rec, f1])

print("\n\n --- Performance Table ---")
# Fixed: format method syntax and variable name (was result, should be results)
print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("Split", "Accuracy", "Precision", "Recall", "F1 Score"))
print("-" * 50)
for row in results:
    print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(row[0], row[1], row[2], row[3], row[4]))


