import pandas as pd
from mlxtend.frequent_patterns import association_rules
from itertools import combinations

transactions = [
    ['M', 'O', 'N', 'K', 'E', 'Y'],
    ['D', 'O', 'N', 'K', 'E', 'Y'],
    ['M', 'A', 'K', 'E'],
    ['M', 'U', 'C', 'K', 'Y'],
    ['C', 'O', 'O', 'K', 'I', 'E']]

# Single-item frequent sets
n = len(transactions)
min_support = 0.6
results = []

# Build vertical DB
vertical_db = {}
for tid, tx in enumerate(transactions):
    for item in tx:
        vertical_db.setdefault(item, set()).add(tid)

# 2. STEP A: Add Single Items (1-itemsets)
frequent_items = {}
for item, tids in vertical_db.items():
    support = len(tids) / n
    if support >= min_support:
        frequent_items[item] = tids
        results.append({'itemsets': frozenset([item]), 'support': support})

# 3. STEP B: Add Pairs (2-itemsets) via Intersection
for item1, item2 in combinations(frequent_items.keys(), 2):
    common_tids = frequent_items[item1] & frequent_items[item2]
    support = len(common_tids) / n
    if support >= min_support:
        results.append({'itemsets': frozenset([item1, item2]), 'support': support})

frequent_itemsets_eclat = pd.DataFrame(results)
print(frequent_itemsets_eclat)

# Generate association rules (min confidence = 0.7)
rules_eclat = association_rules(frequent_itemsets_eclat, metric="confidence", min_threshold=0.7)
print("Association Rules (ECLAT):")
print(rules_eclat[['antecedents', 'consequents', 'support', 'confidence', 'lift']])