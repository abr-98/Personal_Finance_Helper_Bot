import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from warnings import filterwarnings
from utils import generate_advice_final

filterwarnings("ignore")

app = FastAPI(title="Personal Spending Habits API")

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")
cluster = joblib.load("cluster.pkl")
lime_explainer = joblib.load("lime_explainer.pkl")



@app.get("/")
def read_root():
    return {"message": "Welcome to the Personal Spending Habits API"}  


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


@app.post("/analyze")
def potential_savings(user_data: UserData):
    try:
        df = pd.DataFrame([{
            "Age": user_data.Age,
            "Dependents": user_data.Dependents,
            "City_Tier": user_data.City_Tier,
            "Rent": user_data.Rent_As_Percentage_Of_Income,
            "Loan_Repayment": user_data.Loan_Repayment_As_Percentage_Of_Income,
            "Insurance": user_data.Insurance_As_Percentage_Of_Income,
            "Groceries": user_data.Groceries_As_Percentage_Of_Income,
            "Transport": user_data.Transport_As_Percentage_Of_Income,
            "Eating_Out": user_data.Eating_Out_As_Percentage_Of_Income,
            "Entertainment": user_data.Entertainment_As_Percentage_Of_Income,
            "Utilities": user_data.Utilities_As_Percentage_Of_Income,
            "Healthcare": user_data.Healthcare_As_Percentage_Of_Income,
            "Education": user_data.Education_As_Percentage_Of_Income,
            "Miscellaneous": user_data.Miscellaneous_As_Percentage_Of_Income,
            "Disposable_Income": user_data.Disposable_Income_As_Percentage_Of_Income

        }])

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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    

