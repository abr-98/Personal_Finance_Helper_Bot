import pandas as pd
import joblib
from fastmcp import FastMCP
from warnings import filterwarnings

filterwarnings("ignore")

mcp = FastMCP(title="Risk Estimator API")

model = joblib.load("rf_model.pkl")
lime_explainer = joblib.load("lime_explainer.pkl")

@mcp.tool()
def analyze_risk(
    BankruptcyHistory: int,
    LoanAmountToIncome: float,
    PreviousLoanDefaults: int,
    CreditScore: float,
    TotalLiabilitiesToIncome: float,
    Experience: int,
    Age: int,
    EducationLevel: int,
    NetWorthToIncome: float,
    EmploymentStatus_Unemployed: int
) -> dict:

    """
        Analyze potential risk based on user input.

        Parameters:
        - BankruptcyHistory (int): Whether the user has a history of bankruptcy (0 or 1).
        - LoanAmountToIncome (float): The ratio of the loan amount to the user's income.
        - PreviousLoanDefaults (int): The number of previous loan defaults.
        - CreditScore (float): The user's credit score.
        - TotalLiabilitiesToIncome (float): The ratio of total liabilities to the user's income.
        - Experience (int): The number of years of financial experience.
        - Age (int): The age of the user.
        - EducationLevel (int): The education level of the user (e.g., 1 for high school, 2 for bachelor's, etc.).
        - NetWorthToIncome (float): The ratio of net worth to the user's income.
        - EmploymentStatus_Unemployed (int): Whether the user is unemployed (0 or 1).

        Returns:
        - dict: A dictionary containing the potential risk class and an explanation.
    """
    try:
        df = pd.DataFrame([{
            "BankruptcyHistory": BankruptcyHistory,
            "LoanAmountToIncome": LoanAmountToIncome,
            "PreviousLoanDefaults": PreviousLoanDefaults,
            "CreditScore": CreditScore,
            "TotalLiabilitiesToIncome": TotalLiabilitiesToIncome,
            "Experience": Experience,
            "Age": Age,
            "EducationLevel": EducationLevel,
            "NetWorthToIncome": NetWorthToIncome,
            "EmploymentStatus_Unemployed": EmploymentStatus_Unemployed
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
    mcp.run("streamable-http")