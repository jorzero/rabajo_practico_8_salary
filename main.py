from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import joblib
from pydantic import BaseModel

app = FastAPI(
    title="Salary Prediction Model",
    version="1.0.0"
)

# Load the Linear Regression Model
model = joblib.load("model/saved_model.pkl")

# Define a Pydantic model for the input data
class SalaryInput(BaseModel):
    years_experience: float

@app.post("/predict-salary", status_code=200)
async def predict_salary(years_experience: float):
    try:
        prediction = model.predict([[years_experience]])
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"predicted_salary": prediction[0]}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

# To run the server, use the command: uvicorn main:app --reload