import pandas as pd
import joblib
from fastmcp import FastMCP
from warnings import filterwarnings
import json


filterwarnings("ignore")

mcp = FastMCP(title="Financial Trends API")

expect_encoder = joblib.load("Expect_label_encoder.pkl")
gender_encoder = joblib.load("gender_label_encoder.pkl")
Object_encoder = joblib.load("Objective_label_encoder.pkl")
Purpose_encoder = joblib.load("Purpose_label_encoder.pkl")

model = joblib.load("model.pkl")
lime_explainer = joblib.load("explainer.pkl")
cluster = joblib.load("cluster.pkl")
investment_options = pd.read_csv("mean_investment_options.csv")

reason_dict = json.load(open("reason_dict.json", "r"))

COLUMNS = [
    "gender", 
    "Objective", 
    "Expect", 
    "Purpose", 
    "age"]

@mcp.tool()
def analyze_trends(
    gender: str, 
    Objective: str, 
    Expect: str, 
    Purpose: str, 
    age: int) -> dict:

    """
    Analyze financial trends based on user input.

    Parameters:
    - gender (str): The gender of the user.
    - Objective (str): The financial objective of the user.
    - Expect (str): The user's expectations.
    - Purpose (str): The purpose of the financial plan.
    - age (int): The age of the user.

    Returns:
    - dict: A dictionary containing the analysis results.
    """
    try:
        df = pd.DataFrame([{
            gender_encoder.transform([gender])[0],
            Object_encoder.transform([Objective])[0],
            expect_encoder.transform([Expect])[0],
            Purpose_encoder.transform([Purpose])[0],
            age
        }])

        df.reset_index(inplace=True)


        prediction = cluster.predict(df.values)
        pred = int(prediction[0])

        average_investments = investment_options[investment_options["cluster"] == pred].iloc[0]


        average_investments = average_investments.drop("cluster")
        avergage_investments_dict = average_investments.to_dict()

        df.columns = ["gender", "Objective", "Expect", "Purpose", "age"]


        lime_explanation = lime_explainer.explain_instance(df.iloc[0], model.predict_proba, num_features=5)

        keys = reason_dict.keys()

        max_avg_investment = max(avergage_investments_dict, key=lambda x: avergage_investments_dict[x] if x in keys else float('-inf'))
        reason = reason_dict.get(max_avg_investment, "No specific reason available")


        return {
            "average_investment_options": avergage_investments_dict,
            "reason": reason
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run("streamable-http")