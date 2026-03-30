import pandas as pd
import joblib
from fastmcp import FastMCP
from warnings import filterwarnings
from pydantic import BaseModel, Field
import json


filterwarnings("ignore")

mcp = FastMCP("Financial Trends API")

class UserData(BaseModel):
    gender: str = Field(example="Male", description="Male/Female")
    Objective: str = Field(example="Growth", description="Investment objective (Capital Appreciation, Growth, Income)")
    Expect: str = Field(example="high", description="Return expectation (high, medium, low)")
    Purpose: str = Field(example="Wealth Creation", description="Investment purpose (Wealth Creation, Savings for Future, Returns)")
    age: int = Field(example=30)

class ReturnData(BaseModel):
    average_investment_options: dict
    reason: str

expect_encoder = joblib.load("Expect_label_encoder.pkl")
gender_encoder = joblib.load("gender_label_encoder.pkl")
Object_encoder = joblib.load("Objective_label_encoder.pkl")
Purpose_encoder = joblib.load("Purpose_label_encoder.pkl")

model = joblib.load("model.pkl")
lime_explainer = joblib.load("explainer.pkl")
cluster = joblib.load("cluster.pkl")
investment_options = pd.read_csv("mean_investment_options.csv")

reason_dict = json.load(open("reason_dict.json", "r"))


@mcp.tool()
def analyze_trends(
    UserData: UserData
) -> ReturnData:

    """
    _summary_
        "name": "analyze_trends",
        "description": "Analyzes financial trend preferences based on user profile inputs.",
        "input_schema": {
            "gender": "string",
            "Objective": "string",
            "Expect": "string",
            "Purpose": "string",
            "age": "integer"
        },
    Returns:
        output_schema: {
            "average_investment_options": "object",
            "reason": "string"
        }
    """
    try:

        dict_input = {
            "gender": gender_encoder.transform([UserData.gender])[0],
            "Objective": Object_encoder.transform([UserData.Objective])[0],
            "Expect": expect_encoder.transform([UserData.Expect])[0],
            "Purpose": Purpose_encoder.transform([UserData.Purpose])[0],
            "age": UserData.age
        }

        missing = [k for k, v in dict_input.items() if v is None]
        
        if missing:
            return ReturnData(
                average_investment_options={},
                reason=f"Missing fields: {', '.join(missing)}"
            )
        
        df = pd.DataFrame([dict_input])


        prediction = cluster.predict(df.values)
        pred = int(prediction[0])

        average_investments = investment_options[investment_options["cluster"] == pred].iloc[0]


        average_investments = average_investments.drop("cluster")
        avergage_investments_dict = average_investments.to_dict()


        lime_explanation = lime_explainer.explain_instance(df.iloc[0], model.predict_proba, num_features=5)

        keys = reason_dict.keys()

        max_avg_investment = max(avergage_investments_dict, key=lambda x: avergage_investments_dict[x] if x in keys else float('-inf'))
        reason = reason_dict.get(max_avg_investment, "No specific reason available")


        return ReturnData(
            average_investment_options=avergage_investments_dict,
            reason=",".join(lime_explanation.as_list()) + f" | Reason for top investment choice: {reason}"
        )
    except Exception as e:
        return ReturnData(
            average_investment_options={},
            reason=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    mcp.run("streamable-http")