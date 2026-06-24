import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense

df = pd.read_csv(
"backend/data/processed_data.csv"
)

df["batting_team"]=0

X=df[
[
"batting_team",
"current_score",
"wickets",
"overs",
"run_rate",
"pressure",
"momentum"
]
]

y=df["pred_feature"]

scaler=joblib.load(
"backend/model/scaler.pkl"
)

X=scaler.transform(X)

X=np.reshape(
X,
(
X.shape[0],
1,
X.shape[1]
)
)

model=Sequential()

model.add(
SimpleRNN(
64,
input_shape=(1,7)
)
)

model.add(
Dense(1)
)

model.compile(
optimizer="adam",
loss="mse"
)

model.fit(
X,
y,
epochs=40
)

model.save(
"backend/model/rnn_model.keras"
)

print(
"RNN Saved"
)