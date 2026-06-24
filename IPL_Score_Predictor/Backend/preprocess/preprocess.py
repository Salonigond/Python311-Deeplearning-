import pandas as pd

matches=pd.read_csv(
"backend/data/matches.csv"
)

deliveries=pd.read_csv(
"backend/data/deliveries.csv"
)

df=deliveries.copy()

df["current_score"]=(
df.groupby(
"match_id"
)["total_runs"].cumsum()
)

df["wickets"]=(
df.groupby(
"match_id"
)["wicket"].cumsum()
)

df["ball_number"]=(
(df["over"]-1)*6+
df["ball"]
)

df["overs"]=(
df["ball_number"]/6
)

df["run_rate"]=(
df["current_score"]/
df["overs"]
)

df=df.fillna(0)

df.to_csv(
"backend/data/processed_data.csv",
index=False
)

print(
"Preprocessing Complete"
)