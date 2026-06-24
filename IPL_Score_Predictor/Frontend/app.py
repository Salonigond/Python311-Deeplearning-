import streamlit as st
import requests


st.set_page_config(
page_title=
"IPL Predictor",
layout=
"wide"
)


st.title(
"🏏 IPL Score Predictor"
)


team = st.selectbox(
"Batting Team",
[
"MI",
"RCB",
"CSK",
"KKR",
"SRH",
"GT",
"RR",
"LSG",
"PBKS",
"DC"
]
)


score = st.number_input(
"Current Score",
0
)

wickets = st.slider(
"Wickets",
0,
10
)

overs = st.slider(
"Overs",
0.1,
20.0
)

run_rate = (
score/
max(
overs,
1
)
)

pressure = (
wickets*2
)

momentum = (
run_rate*1.5
)


if st.button(
"Predict"
):

    payload = {

    "team":
    team,

    "current_score":
    score,

    "wickets":
    wickets,

    "overs":
    overs,

    "run_rate":
    run_rate,

    "pressure":
    pressure,

    "momentum":
    momentum
    }

    r = requests.post(
    "http://127.0.0.1:8000/predict",
    json=payload
    )

    result = r.json()

    st.success(
        f"Predicted Score: "
        f"{result['predicted_score']}"
    )

    st.metric(
        "Expected Score",
        result[
        "predicted_score"
        ]
    )