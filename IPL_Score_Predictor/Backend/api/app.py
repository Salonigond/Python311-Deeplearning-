from fastapi import FastAPI

from pydantic import BaseModel

from Backend.model.predict import predict_score

app = FastAPI()


class MatchInput(
BaseModel
):

    team:str
    current_score:int
    wickets:int
    overs:float
    run_rate:float
    pressure:float
    momentum:float


@app.get("/")
def home():

    return {
        "message":
        "IPL Predictor Running"
    }


@app.post(
"/predict"
)

def predict(
data:MatchInput
):

    score = predict_score(
        data.team,
        data.current_score,
        data.wickets,
        data.overs,
        data.run_rate,
        data.pressure,
        data.momentum
    )

    return {
        "predicted_score":
        score
    }