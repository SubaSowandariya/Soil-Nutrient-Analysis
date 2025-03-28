import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from groq import Groq 
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
if api_key is None:
    raise ValueError("API key not found. Please set the GROQ_API_KEY environment variable.")
client = Groq(api_key=api_key)

# Load datasets
macro_df = pd.read_csv("Cleaned_Macronutrients.csv")
micro_df = pd.read_csv("Cleaned_Micronutrients.csv")

# Merge both datasets on 'village'
data = pd.merge(macro_df, micro_df, on='village')

# Soil Deficiency Analysis
deficiency_columns = [col for col in data.columns if "deficient" in col]
data["Total_Deficiencies"] = data[deficiency_columns].sum(axis=1)

# New Water Requirement Index (WRI) based on summation formula
data["WRI"] = data[deficiency_columns].sum(axis=1)

# Normalize WRI
scaler = StandardScaler()
data["WRI_Normalized"] = scaler.fit_transform(data[["WRI"]])

# Save the StandardScaler model
joblib.dump(scaler, "scaler_model.pkl")

# Clustering Villages into Water Demand Zones
kmeans = KMeans(n_clusters=3, random_state=42)
data["Irrigation_Zone"] = kmeans.fit_predict(data[["WRI_Normalized"]])

# Save the KMeans model
joblib.dump(kmeans, "kmeans_model.pkl")

# Train ML Model for Predicting Water Requirements
X = data.drop(columns=["village", "WRI", "WRI_Normalized", "Irrigation_Zone"])
y = data["WRI"]
model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X, y)

# Save the trained LightGBM model
joblib.dump(model, "lgbm_model.pkl")

# Predict Water Requirement
data["Predicted_Water_Requirement"] = model.predict(X)

# Calculate and print model accuracy
accuracy = r2_score(y, data["Predicted_Water_Requirement"])
print(f"Model Accuracy (R² Score): {accuracy:.4f}")

# Generate AI-Based Smart Irrigation Plans
def irrigation_plan(wri):
    if wri < data["WRI"].quantile(0.33):
        return "Low Water Zone - Use drip irrigation"
    elif wri < data["WRI"].quantile(0.66):
        return "Medium Water Zone - Irrigate every 3 days"
    else:
        return "High Water Zone - Use rainwater harvesting & sprinklers"

data["Irrigation_Schedule"] = data["Predicted_Water_Requirement"].apply(irrigation_plan)

# Function to generate AI insights using Groq API
def generate_ai_insights(village_data):
    prompt = (
        f"Provide detailed irrigation recommendations for a village with water requirement index {village_data['Predicted_Water_Requirement']}. "
        f"Suggest possible crops that can be grown in this irrigation zone, appropriate irrigation methods, "
        f"the estimated irrigation time needed for the soil and crops, and effective water management techniques."
    )
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",  # Use the desired Groq model
    )
    return response.choices[0].message.content

# Get user input and display irrigation plan
village_name = input("Enter village name: ")
if village_name not in data["village"].values:
    print("Error: Village not found.")
else:
    village_data = data[data["village"] == village_name].iloc[0]
    irrigation_schedule = village_data["Irrigation_Schedule"]
    ai_insights = generate_ai_insights(village_data)
    
    print(f"\nVillage: {village_name}")
    print(f"Irrigation Zone: {int(village_data['Irrigation_Zone'])}")
    print(f"Predicted Water Requirement: {float(village_data['Predicted_Water_Requirement'])}")
    print(f"Irrigation Schedule: {irrigation_schedule}")
    print(f"AI Suggestions: {ai_insights}")
