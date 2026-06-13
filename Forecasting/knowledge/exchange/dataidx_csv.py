import pandas as pd
import torch

date2idx = torch.load("knowledge/exchange/itr0/date2idx.pt")

df = pd.DataFrame(
    [(k, v) for k, v in date2idx.items()],
    columns=["date", "idx"]
)

# 可选：按日期排序，方便肉眼检查
df = df.sort_values("date").reset_index(drop=True)

df.to_csv(
    "knowledge/exchange/itr0/date2idx.csv",
    index=False
)

print(df.head())
print(df.tail())
print("Total dates:", len(df))
