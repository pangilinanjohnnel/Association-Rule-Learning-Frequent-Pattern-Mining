import pandas as pd
from prefixspan import PrefixSpan


data={"Transaction_ID": ["T001", "T002", "T003", "T004", "T005", "T006", "T007", "T008", "T009"],
"Customer_ID":["C001", "C001", "C002", "C003", "C001", "C002", "C002", "C003", "C001"],
"Item_ID":["I105", "I204", "I105", "I301", "I401", "I105", "I204", "I301", "I105"],
"Item_Name":["Coffee", "Bread", "Coffee", "Shampoo", "Milk", "Coffee", "Bread", "Shampoo", "Coffee"],
"Purchase_Timestamp":["2025-04-01 08:35:00", "2025-04-01 08:36:00", "2025-04-01 09:00:00",
                      "2025-04-02 14:15:00", "2025-04-03 08:30:00", "2025-04-03 09:01:00",
                      "2025-04-03 09:02:00", "2025-04-06 14:20:00", "2025-04-10 08:35:00"]}

df = pd.DataFrame(data)

#CONVERT SORT DATETIME
df["Purchase_Timestamp"] = pd.to_datetime(df["Purchase_Timestamp"])
df = df.sort_values(["Customer_ID", "Purchase_Timestamp"])

#GROUPBY
sequences = df.groupby("Customer_ID")["Item_Name"].apply(list).tolist()
print(sequences)

#FREQUENT PATTERNS
ps = PrefixSpan(sequences)
patterns = ps.frequent(2)
print(patterns)

