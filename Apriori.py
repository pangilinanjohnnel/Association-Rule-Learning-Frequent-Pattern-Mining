import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

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

# Frequent itemsets with min support 0.6
frequent_itemsets_apriori = apriori(df, min_support=0.6, use_colnames=True)

print("Frequent Itemsets (Apriori):")
print(frequent_itemsets_apriori)

# Association rules with min confidence 0.7
rules_apriori = association_rules(
    frequent_itemsets_apriori,
    metric="confidence",
    min_threshold=0.7)
print("Association Rules (Apriori):")
print(rules_apriori[['antecedents', 'consequents', 'support', 'confidence', 'lift']])


