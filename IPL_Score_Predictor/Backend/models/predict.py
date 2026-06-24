import numpy as np
import joblib

from tensorflow.keras.models import load_model


MODEL_PATH = (
"backend/model/lstm_model.keras"
)

SCALER_PATH = (
"backend/model/scaler.pkl"
)


model = load_model(
MODEL_PATH
)

scaler = joblib.load(
SCALER_PATH
)


TEAM_MAP = {
"MI":0,
"RCB":1,
"CSK":2,
"KKR":3,
"SRH":4,
"GT":5,
"RR":6,
"LSG":7,
"PBKS":8,
"DC":9
}


def predict_score(
team,
current_score,
wickets,
overs,
run_rate,
pressure,
momentum
):

    team = TEAM_MAP.get(
        team,
        0
    )

    x = [[
        team,
        current_score,
        wickets,
        overs,
        run_rate,
        pressure,
        momentum
    ]]

    x = scaler.transform(
        x
    )

    x = np.reshape(
        x,
        (
            x.shape[0],
            1,
            x.shape[1]
        )
    )

    pred = model.predict(
        x,
        verbose=0
    )

    return round(
        float(
            pred[0][0]
        )
    )