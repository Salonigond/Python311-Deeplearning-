import pandas as pd

df=pd.read_csv(
"backend/data/processed_data.csv"
)

df["pressure"]=(
df["wickets"]*2
)

df["momentum"]=(
df["run_rate"]*
1.5
)

df["pred_feature"]=(
df["current_score"]+
df["momentum"]
)

df.to_csv(
"backend/data/processed_data.csv",
index=False
)

print(
"Features Added"
)