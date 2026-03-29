import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from warnings import filterwarnings

filterwarnings("ignore")

app = FastAPI(title="Risk Estimator API")

model = joblib.load("rf_model.pkl")
lime_explainer = joblib.load("lime_explainer.pkl")



@app.get("/")
def read_root():
    return {"message": "Welcome to the Risk Estimator API"} 

class UserData(BaseModel):
    BankruptcyHistory: int = Field(example=0)
    LoanAmountToIncome: float = Field(example=0.3)
    PreviousLoanDefaults: int = Field(example=0)
    CreditScore: float = Field(example=700)
    TotalLiabilitiesToIncome: float = Field(example=0.5)
    Experience: int = Field(example=5)
    Age: int = Field(example=30)
    EducationLevel: int = Field(example=3)
    NetWorthToIncome: float = Field(example=1.5)
    EmploymentStatus_Unemployed: int = Field(example=0)

@app.post("/analyze")
def analyze_risk(user_data: UserData):
    try:
        df = pd.DataFrame([{
            "BankruptcyHistory": user_data.BankruptcyHistory,
            "LoanAmountToIncome": user_data.LoanAmountToIncome,
            "PreviousLoanDefaults": user_data.PreviousLoanDefaults,
            "CreditScore": user_data.CreditScore,
            "TotalLiabilitiesToIncome": user_data.TotalLiabilitiesToIncome,
            "Experience": user_data.Experience,
            "Age": user_data.Age,
            "EducationLevel": user_data.EducationLevel,
            "NetWorthToIncome": user_data.NetWorthToIncome,
            "EmploymentStatus_Unemployed": user_data.EmploymentStatus_Unemployed
        }])
        df_val = df.values
        prediction = model.predict(df_val)[0]
        pred = int(prediction)

        explanation = lime_explainer.explain_instance(df.iloc[0], model.predict_proba, num_features=5)

        mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        risk_category = mapping.get(pred, "Unknown")
        return {
            "potential_risk_class": risk_category,
            "explanation": explanation.as_list()
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)