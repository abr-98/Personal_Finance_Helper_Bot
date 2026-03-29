import pandas as pd
import joblib
from fastmcp import FastMCP
from warnings import filterwarnings
from utils import generate_advice_final


filterwarnings("ignore")

mcp = FastMCP(title="Personal Spending Habits API")

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")
cluster = joblib.load("cluster.pkl")
lime_explainer = joblib.load("lime_explainer.pkl")

columns = [
    "Age",
    "Dependents",
    "City_Tier",
    "Rent_As_Percentage_Of_Income",
    "Loan_Repayment_As_Percentage_Of_Income",
    "Insurance_As_Percentage_Of_Income",
    "Groceries_As_Percentage_Of_Income",
    "Transport_As_Percentage_Of_Income",
    "Eating_Out_As_Percentage_Of_Income",
    "Entertainment_As_Percentage_Of_Income",
    "Utilities_As_Percentage_Of_Income",
    "Healthcare_As_Percentage_Of_Income",
    "Education_As_Percentage_Of_Income",
    "Miscellaneous_As_Percentage_Of_Income",
    "Disposable_Income_As_Percentage_Of_Income"
]

@mcp.tool()
def potential_savings(
    Age: int,
    Dependents: int,
    City_Tier: int,
    Rent_As_Percentage_Of_Income: float,
    Loan_Repayment_As_Percentage_Of_Income: float,
    Insurance_As_Percentage_Of_Income: float,
    Groceries_As_Percentage_Of_Income: float,
    Transport_As_Percentage_Of_Income: float,
    Eating_Out_As_Percentage_Of_Income: float,
    Entertainment_As_Percentage_Of_Income: float,
    Utilities_As_Percentage_Of_Income: float,
    Healthcare_As_Percentage_Of_Income: float,
    Education_As_Percentage_Of_Income: float,
    Miscellaneous_As_Percentage_Of_Income: float,
    Disposable_Income_As_Percentage_Of_Income: float
) -> dict:

    """
        Analyze potential savings based on user input.

        Parameters:
        - Age (int): The age of the user.
        - Dependents (int): The number of dependents the user has.
        - City_Tier (int): The city tier of the user's residence (1, 2, or 3).
        - Rent_As_Percentage_Of_Income (float): Rent as a percentage of income.
        - Loan_Repayment_As_Percentage_Of_Income (float): Loan repayment as a percentage of income.
        - Insurance_As_Percentage_Of_Income (float): Insurance as a percentage of income.
        - Groceries_As_Percentage_Of_Income (float): Groceries as a percentage of income.
        - Transport_As_Percentage_Of_Income (float): Transport as a percentage of income.
        - Eating_Out_As_Percentage_Of_Income (float): Eating out as a percentage of income.
        - Entertainment_As_Percentage_Of_Income (float): Entertainment as a percentage of income
        - Utilities_As_Percentage_Of_Income (float): Utilities as a percentage of income.
        - Healthcare_As_Percentage_Of_Income (float): Healthcare as a percentage of income.
        - Education_As_Percentage_Of_Income (float): Education as a percentage of income.
        - Miscellaneous_As_Percentage_Of_Income (float): Miscellaneous expenses as a percentage of income.
        - Disposable_Income_As_Percentage_Of_Income (float): Disposable income as a percentage of income.
        
        Returns:
        - dict: A dictionary containing the potential savings class, explanation, average behavior summary for similar
            users, and personalized advice.
        """
    try:
        df = pd.DataFrame([{
            "Age": Age,
            "Dependents": Dependents,
            "City_Tier": City_Tier,
            "Rent": Rent_As_Percentage_Of_Income,
            "Loan_Repayment": Loan_Repayment_As_Percentage_Of_Income,
            "Insurance": Insurance_As_Percentage_Of_Income,
            "Groceries": Groceries_As_Percentage_Of_Income,
            "Transport": Transport_As_Percentage_Of_Income,
            "Eating_Out": Eating_Out_As_Percentage_Of_Income,
            "Entertainment": Entertainment_As_Percentage_Of_Income,
            "Utilities": Utilities_As_Percentage_Of_Income,
            "Healthcare": Healthcare_As_Percentage_Of_Income,
            "Education": Education_As_Percentage_Of_Income,
            "Miscellaneous": Miscellaneous_As_Percentage_Of_Income,
            "Disposable_Income": Disposable_Income_As_Percentage_Of_Income
        }])

        df = df[columns]

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


        return {"potential_savings_class": category,
                "explanation": explanation.as_list(),
                "average_behavior_summary_for_similar_users": behavior_summary,
                "advice": advice
                }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run("streamable-http")

