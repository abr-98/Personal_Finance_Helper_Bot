import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from warnings import filterwarnings
import json

filterwarnings("ignore")

app = FastAPI(title="Financial Trends API")

expect_encoder = joblib.load("Expect_label_encoder.pkl")
gender_encoder = joblib.load("gender_label_encoder.pkl")
Object_encoder = joblib.load("Objective_label_encoder.pkl")
Purpose_encoder = joblib.load("Purpose_label_encoder.pkl")

model = joblib.load("model.pkl")
lime_explainer = joblib.load("explainer.pkl")
cluster = joblib.load("cluster.pkl")
investment_options = pd.read_csv("mean_investment_options.csv")

reason_dict = json.load(open("reason_dict.json", "r"))


@app.get("/")
def read_root():
    return {"message": "Welcome to the Financial Trends API"}  


class UserData(BaseModel):
    gender: str = Field(example="Male", description="Male/Female")
    Objective: str = Field(example="Growth", description="Investment objective (Capital Appreciation, Growth, Income)")
    Expect: str = Field(example="high", description="Return expectation (high, medium, low)")
    Purpose: str = Field(example="Wealth Creation", description="Investment purpose (Wealth Creation, Savings for Future, Returns)")
    age: int = Field(example=30)


@app.post("/analyze")
def analyze_trends(user_data: UserData):
    try:
        df = pd.DataFrame([{
            gender_encoder.transform([user_data.gender])[0],
            Object_encoder.transform([user_data.Objective])[0],
            expect_encoder.transform([user_data.Expect])[0],
            Purpose_encoder.transform([user_data.Purpose])[0],
            user_data.age
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)