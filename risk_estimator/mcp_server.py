import pandas as pd
import joblib
from fastmcp import FastMCP
from warnings import filterwarnings
from pydantic import BaseModel, Field

filterwarnings("ignore")

mcp = FastMCP("Risk Estimator API")

model = joblib.load("rf_model.pkl")
lime_explainer = joblib.load("lime_explainer.pkl")

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

class ReturnData(BaseModel):
    potential_risk_class: str
    explanation: list


@mcp.tool()
def analyze_risk(
    UserData: UserData
) -> ReturnData:

    """
        _summary_
            "name": "analyze_risk",
            "description": "Analyzes loan default risk based on financial and demographic indicators.",
            "input_schema": {
                "BankruptcyHistory": "integer",
                "LoanAmountToIncome": "float",
                "PreviousLoanDefaults": "integer",
                "CreditScore": "float",
                "TotalLiabilitiesToIncome": "float",
                "Experience": "integer",
                "Age": "integer",
                "EducationLevel": "integer",
                "NetWorthToIncome": "float",
                "EmploymentStatus_Unemployed": "integer"
            },
        Returns:
            output_schema: {
                "potential_risk_class": "string",
                "explanation": "array"
            }
    """
    try:

        dict_input = {
            "BankruptcyHistory": UserData.BankruptcyHistory,
            "LoanAmountToIncome": UserData.LoanAmountToIncome,
            "PreviousLoanDefaults": UserData.PreviousLoanDefaults,
            "CreditScore": UserData.CreditScore,
            "TotalLiabilitiesToIncome": UserData.TotalLiabilitiesToIncome,
            "Experience": UserData.Experience,
            "Age": UserData.Age,
            "EducationLevel": UserData.EducationLevel,
            "NetWorthToIncome": UserData.NetWorthToIncome,
            "EmploymentStatus_Unemployed": UserData.EmploymentStatus_Unemployed
        }

        missing = [k for k, v in dict_input.items() if v is None]

        if missing:
            return ReturnData(
                potential_risk_class="Unknown",
                explanation=[f"Missing fields: {', '.join(missing)}"]
            )

        df = pd.DataFrame([dict_input])
        df_val = df.values
        prediction = model.predict(df_val)[0]
        pred = int(prediction)

        explanation = lime_explainer.explain_instance(df.iloc[0], model.predict_proba, num_features=5)

        mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        risk_category = mapping.get(pred, "Unknown")
        return ReturnData(
            potential_risk_class=risk_category,
            explanation=explanation.as_list()
        )
    except Exception as e:
        return ReturnData(
            potential_risk_class="Unknown",
            explanation=[f"Error: {str(e)}"]
        )

if __name__ == "__main__":
    mcp.run("streamable-http")