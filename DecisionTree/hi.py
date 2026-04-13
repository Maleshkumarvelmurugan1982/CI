import csv, math
from collections import Counter

def load_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)

def entropy(rows, target):
    counts = Counter(r[target] for r in rows)
    n = len(rows)
    H = 0
    print(f"\n  Classes: {dict(counts)}")
    print(f"  Formula: H = -Σ p(c) × log₂(p(c))")
    for cls, cnt in counts.items():
        p = cnt / n
        term = -p * math.log2(p)
        print(f"    p({cls}) = {cnt}/{n} = {p:.4f}  →  -{p:.4f} × log₂({p:.4f}) = {term:.4f}")
        H += term
    print(f"  → H = {H:.4f}")
    return H

def info_gain(rows, attr, target, parent_H):
    groups = {}
    for r in rows:
        groups.setdefault(r[attr], []).append(r)

    print(f"\n  Attribute: '{attr}'")
    weighted = 0
    for val, subset in groups.items():
        H = entropy(subset, target)
        w = len(subset) / len(rows)
        print(f"    {attr}={val}: weight={w:.4f}, H={H:.4f}  →  {w:.4f} × {H:.4f} = {w*H:.4f}")
        weighted += w * H

    IG = parent_H - weighted
    print(f"  IG({attr}) = {parent_H:.4f} - {weighted:.4f} = {IG:.4f}")
    return IG

def find_root(path, target):
    rows = load_csv(path)
    features = [k for k in rows[0] if k != target]

    print("=" * 50)
    print(f"STEP 1: Parent entropy for '{target}'")
    print("=" * 50)
    parent_H = entropy(rows, target)

    print("\n" + "=" * 50)
    print("STEP 2: Information gain per attribute")
    print("=" * 50)
    gains = {attr: info_gain(rows, attr, target, parent_H) for attr in features}

    root = max(gains, key=gains.get)
    print("\n" + "=" * 50)
    print("STEP 3: Summary")
    print("=" * 50)
    for attr, ig in sorted(gains.items(), key=lambda x: -x[1]):
        marker = " ← ROOT" if attr == root else ""
        print(f"  {attr:20s}: IG = {ig:.4f}{marker}")
    print(f"\n✓ Root node = '{root}' (IG = {gains[root]:.4f})")
    return root

# --- Run ---
find_root("data.csv", "PlayTennis")
