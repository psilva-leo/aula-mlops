import pandas as pd
from fastapi import FastAPI, HTTPException

from model import Model
from schemas import DataRequest

app = FastAPI()
model = Model()


@app.post("/infer")
async def infer(request: DataRequest):
    try:
        payload = request.model_dump(by_alias=True)
        df = pd.DataFrame(payload, index=[0])
        prediction = model.predict(df)

        return {"house_pricing": prediction.tolist()[0]}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
