from enum import Enum, IntEnum
from typing import List

import fastapi
import pandas as pd
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config_loader import APIConfigLoader, ModelConfigLoader

# Load configs from defaults.yml
cfg_model = ModelConfigLoader()
cfg_api = APIConfigLoader()

OPERA_DATA = cfg_api.models["FlightModel"]["OPERA"]
MES_DATA = cfg_api.models["FlightModel"]["MES"]
TIPOVUELO_DATA = cfg_api.models["FlightModel"]["TIPOVUELO"]


# Create dynamics enums values got from default.yml values.
OperatorEnum = Enum("OperatorEnum", {k: k for k in OPERA_DATA})
MESEnum = IntEnum("MESEnum", {str(k): k for k in MES_DATA})
TIPOVUELOEnum = Enum("TIPOVUELOEnum", {k: k for k in TIPOVUELO_DATA})


# Models could be created in a different layer, I left it here for simplicity
class Flight(BaseModel):

    OPERA: OperatorEnum
    TIPOVUELO: TIPOVUELOEnum
    MES: MESEnum

    class Config:
        use_enum_values = True


class Flights(BaseModel):
    flights: List[Flight]

    class Config:
        use_enum_values = True


class ResponseModel(BaseModel):
    predict: List[int]


# Api instantiation
app = fastapi.FastAPI()


@app.exception_handler(RequestValidationError)
async def unicorn_exception_handler(request: Request, exc: RequestValidationError):
    """
    This is a hack to comply with the code rules requested in test_api.py.
    I'm changing 422 code throw when a entity validation fail to 400
    for all requests failing with .
    """

    details = exc.errors()
    modified_details = []
    # Replace 'msg' with 'message' for each error
    for error in details:
        modified_details.append({"loc": error["loc"], "message": error["msg"]})
    return JSONResponse(
        status_code=fastapi.status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder({"detail": modified_details}),
    )


from .model import DelayModel

model = DelayModel()


@app.get("/health", status_code=200)
async def get_health() -> dict:

    return {"status": "OK"}


def preprocess_model_request(data: Flights):
    """
    Preprocess input data model in the way ML model recive inputs for prediction
    """
    df = pd.DataFrame(data=data.dict()["flights"])

    # Call same ML model method used during training preprocessing of features
    features = model.preprocess_features(df)

    # Add all missing categorical features not included with value 0
    populated_input_colums = set(features.columns)
    expected_model_columns = set(cfg_model.train_features_name)

    missing_columns = expected_model_columns.difference(populated_input_colums)
    for missing_column in missing_columns:
        features.loc[:, missing_column] = 0

    # Filter only the features and the right order needed for the model prediction
    filtered_features = features[cfg_model.train_features_name]

    return filtered_features


@app.post("/predict", status_code=200, response_model=ResponseModel)
async def post_predict(flights: Flights) -> dict:

    data_prepared = preprocess_model_request(flights)
    predictions = model.predict(data_prepared)
    response_data = ResponseModel(**dict(predict=predictions))

    return JSONResponse(content=response_data.dict())
