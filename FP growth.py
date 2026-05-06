import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

transactions = [
    ['M', 'O', 'N', 'K', 'E', 'Y'],
    ['D', 'O', 'N', 'K', 'E', 'Y'],
    ['M', 'A', 'K', 'E'],
    ['M', 'U', 'C', 'K', 'Y'],
    ['C', 'O', 'O', 'K', 'I', 'E']]

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)

# FP-Growth frequent itemsets
frequent_itemsets_fp = fpgrowth(df, min_support=0.6, use_colnames=True)
print("Frequent Itemsets (FP-Growth):")
print(frequent_itemsets_fp)

# Association rules
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.7)
print("Association Rules (FP-Growth):")
print(rules_fp[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

