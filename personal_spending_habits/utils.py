import pandas as pd


def generate_advice_final(user_data):
    df_input = pd.read_csv("pivot_analysis.csv")

    user = user_data.iloc[0]

    recs = get_recommendations(user, df_input)
    advice = generate_advice(recs)

    return advice



def get_recommendations(user_row, pivot_df, top_n=3):
    # Extract user group
    key = (
        user_row["cluster"],
        user_row["Dependents"],
        user_row["City_Tier"]
    )
    
    group_data = pivot_df[
        (pivot_df["cluster"] == key[0]) &
        (pivot_df["Dependents"] == key[1]) &
        (pivot_df["City_Tier"] == key[2])
    ]
    
    if group_data.empty:
        return ["Not enough similar users for comparison"]
    
    row = group_data.iloc[0]
    
    spending_cols = [
    "Rent", "Loan_Repayment", "Insurance", "Groceries",
    "Transport", "Eating_Out", "Entertainment", "Utilities",
    "Healthcare", "Education", "Miscellaneous"
    ]
    recommendations = []
    
    for col in spending_cols:
        diff = row[f"{col}_diff"]
        user_val = user_row[col]
        high_val = row[f"{col}_1"]
        
        # Only suggest if user is worse than high savers
        if diff < 0 and user_val > high_val:
            recommendations.append((col, abs(diff), user_val - high_val))
    
    # Sort by impact
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    
    return recommendations[:top_n]

def generate_advice(recommendations):
    advice = []
    
    for col, impact, gap in recommendations:
        advice.append(
            f"Reduce {col.replace('_', ' ')} by ~{round(gap, 2)}% "
            f"(people like you spend {round(impact, 2)}% less here)"
        )
    
    return advice