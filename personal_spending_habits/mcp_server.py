import pandas as pd
import joblib
from fastmcp import FastMCP
from warnings import filterwarnings
from utils import generate_advice_final
from pydantic import BaseModel, Field


filterwarnings("ignore")

mcp = FastMCP("Personal Spending Habits API")

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")
cluster = joblib.load("cluster.pkl")
lime_explainer = joblib.load("lime_explainer.pkl")


class UserData(BaseModel):
    Age: int = Field(example=30)
    Dependents: int = Field(example=2)
    City_Tier: int = Field(example=2, description="City tier (1, 2, or 3), 3 for rural areas and 1 for metropolitan areas")
    Rent_As_Percentage_Of_Income: float = Field(example=30.5)
    Loan_Repayment_As_Percentage_Of_Income: float = Field(example=15.0)
    Insurance_As_Percentage_Of_Income: float = Field(example=10.0)
    Groceries_As_Percentage_Of_Income: float = Field(example=20.0)
    Transport_As_Percentage_Of_Income: float = Field(example=5.0)	
    Eating_Out_As_Percentage_Of_Income: float = Field(example=10.0)
    Entertainment_As_Percentage_Of_Income: float = Field(example=5.0)	
    Utilities_As_Percentage_Of_Income: float = Field(example=10.0)
    Healthcare_As_Percentage_Of_Income: float = Field(example=5.0)	
    Education_As_Percentage_Of_Income: float = Field(example=10.0)	
    Miscellaneous_As_Percentage_Of_Income: float = Field(example=5.0)	
    Disposable_Income_As_Percentage_Of_Income: float = Field(example=15.0)

class ReturnData(BaseModel):
    potential_savings_class: str
    explanation: list
    average_behavior_summary_for_similar_users: dict
    advice: str

@mcp.tool()
def potential_savings(
    UserData: UserData
) -> ReturnData:

    """
        _summary_
            "name": "potential_savings",
            "description": "Estimates savings potential and provides spending behavior insights with recommendations.",
            "input_schema": {
                "Age": "integer",
                "Dependents": "integer",
                "City_Tier": "integer",
                "Rent_As_Percentage_Of_Income": "float",
                "Loan_Repayment_As_Percentage_Of_Income": "float",
                "Insurance_As_Percentage_Of_Income": "float",
                "Groceries_As_Percentage_Of_Income": "float",
                "Transport_As_Percentage_Of_Income": "float",
                "Eating_Out_As_Percentage_Of_Income": "float",
                "Entertainment_As_Percentage_Of_Income": "float",
                "Utilities_As_Percentage_Of_Income": "float",
                "Healthcare_As_Percentage_Of_Income": "float",
                "Education_As_Percentage_Of_Income": "float",
                "Miscellaneous_As_Percentage_Of_Income": "float",
                "Disposable_Income_As_Percentage_Of_Income": "float"
            },
        Returns:
            output_schema: {
                "potential_savings_class": "string",
                "explanation": "array",
                "average_behavior_summary_for_similar_users": "object",
                "advice": "string"
            }
        """
    try:

        dict_inp = {
            "Age": UserData.Age,
            "Dependents": UserData.Dependents,
            "City_Tier": UserData.City_Tier,
            "Rent": UserData.Rent_As_Percentage_Of_Income,
            "Loan_Repayment": UserData.Loan_Repayment_As_Percentage_Of_Income,
            "Insurance": UserData.Insurance_As_Percentage_Of_Income,
            "Groceries": UserData.Groceries_As_Percentage_Of_Income,
            "Transport": UserData.Transport_As_Percentage_Of_Income,
            "Eating_Out": UserData.Eating_Out_As_Percentage_Of_Income,
            "Entertainment": UserData.Entertainment_As_Percentage_Of_Income,
            "Utilities": UserData.Utilities_As_Percentage_Of_Income,
            "Healthcare": UserData.Healthcare_As_Percentage_Of_Income,
            "Education": UserData.Education_As_Percentage_Of_Income,
            "Miscellaneous": UserData.Miscellaneous_As_Percentage_Of_Income,
            "Disposable_Income": UserData.Disposable_Income_As_Percentage_Of_Income
        }

        missing = [k for k, v in dict_inp.items() if v is None]
        
        if missing:
            return ReturnData(
                potential_savings_class="Unknown",
                explanation=[],
                average_behavior_summary_for_similar_users={},
                advice=f"Missing fields: {', '.join(missing)}"
            )

        df = pd.DataFrame([dict_inp])

        df_val = df.values
        prediction = model.predict(df_val)[0]
        pred = int(prediction)
        
        category_mapping = {"Very Low": 0, "Low": 1, "Medium": 2, "High": 3, "Very High": 4}
        category_mapping_reverse = {v: k for k, v in category_mapping.items()}
        category = category_mapping_reverse.get(pred, "Unknown")

        explanation = lime_explainer.explain_instance(df.iloc[0], model.predict_proba, num_features=5)

        columns_to_cluster = [
            "Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport",
            "Eating_Out", "Entertainment", "Utilities", "Healthcare",
            "Education", "Miscellaneous", "Disposable_Income"
        ]

        df_cluster = df[columns_to_cluster]
        df_cluster_scaled = scaler.transform(df_cluster)

        cluster_label = cluster.predict(df_cluster_scaled)[0]

        df_summ = pd.read_csv("cluster_summary.csv")


        behavior_summary = df_summ.iloc[cluster_label][:].to_dict()

        df["cluster"] = cluster_label

        print(df.head())

        advice = generate_advice_final(df)


        return ReturnData(
            potential_savings_class=category,
            explanation=explanation.as_list(),
            average_behavior_summary_for_similar_users=behavior_summary,
            advice=" ".join(advice)
        )

    except Exception as e:
        return ReturnData(
            potential_savings_class="Unknown",
            explanation=[f"Error: {str(e)}"],
            average_behavior_summary_for_similar_users={},
            advice=""
        )

if __name__ == "__main__":
    mcp.run("streamable-http")

