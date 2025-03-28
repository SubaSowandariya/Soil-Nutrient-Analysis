import os
import pandas as pd
import numpy as np
import catboost as cb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# Set up Groq API
api_key = os.getenv("GROQ_API_KEY")
if api_key is None:
    raise ValueError("API key not found. Please set the GROQ_API_KEY environment variable.")
client = Groq(api_key=api_key)

# Function to generate AI insights using Groq API
def generate_ai_insights(prompt):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return response.choices[0].message.content

# Load dataset
df_macro = pd.read_csv("Processed_Macronutrients.csv")
df_micro = pd.read_csv("Processed_Micronutrients.csv")

# Merge datasets on Village column
df = pd.merge(df_macro, df_micro, on="Village")

# Define target variable (Deficiency Status based on SDI threshold)
def calculate_sdi(df, sufficiency_cols, deficiency_cols):
    total_sufficient = df[sufficiency_cols].sum(axis=1)
    total_deficient = df[deficiency_cols].sum(axis=1)
    return total_deficient / (total_sufficient + total_deficient)

macro_sufficiency_cols = ["Nitrogen - High", "Phosphorous - High", "Potassium - High", "OC - High", "EC - Non Saline", "pH - Neutral"]
macro_deficiency_cols = ["Nitrogen - Low", "Phosphorous - Low", "Potassium - Low", "OC - Low", "EC - Saline", "pH - Acidic", "pH - Alkaline"]

micro_sufficiency_cols = ["Copper - Sufficient", "Boron - Sufficient", "S - Sufficient", "Fe - Sufficient", "Zn - Sufficient", "Mn - Sufficient"]
micro_deficiency_cols = ["Copper - Deficient", "Boron - Deficient", "S - Deficient", "Fe - Deficient", "Zn - Deficient", "Mn - Deficient"]

# Compute SDI for training
df["SDI_Macro"] = calculate_sdi(df, macro_sufficiency_cols, macro_deficiency_cols)
df["SDI_Micro"] = calculate_sdi(df, micro_sufficiency_cols, micro_deficiency_cols)
df["SDI_Avg"] = (df["SDI_Macro"] + df["SDI_Micro"]) / 2
df["Deficiency_Status"] = (df["SDI_Avg"] > 0.4).astype(int)  # 1 = Deficient, 0 = Not Deficient

# Prepare data for CatBoost
features = macro_sufficiency_cols + macro_deficiency_cols + micro_sufficiency_cols + micro_deficiency_cols
X = df[features].values
y = df["Deficiency_Status"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Label encode villages
df["Village_Encoded"] = LabelEncoder().fit_transform(df["Village"])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train CatBoost model
model = cb.CatBoostClassifier(iterations=500, learning_rate=0.05, depth=8, loss_function='Logloss', verbose=100)
model.fit(X_train, y_train)

# ✅ **Save the trained CatBoost model**  
model.save_model("catboost_model.cbm")  # 🔹 **This line ensures the model is saved for `app.py`**  

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"CatBoost Model Accuracy: {accuracy:.2f}")

# Get user input
village = input("Enter the village name: ")
village_index = df[df["Village"] == village].index.tolist()

if not village_index:
    print("Village not found in dataset.")
else:
    village_features = X[village_index]
    prediction = model.predict(village_features)[0]
    sdi_avg = df.loc[village_index, "SDI_Avg"].values[0]
    deficiency_status = "Deficient" if prediction == 1 else "Not Deficient"

    # Provide recommendations
    if deficiency_status == "Deficient":
        resolution = "Apply balanced fertilizers based on deficiency levels. Use nitrogen-rich fertilizers for better crop yield. Suggested crops: maize, legumes."
    else:
        resolution = "Soil is healthy. Maintain current agricultural practices."

    # Generate AI insights
    prompt = f"Soil deficiency analysis for {village}: Deficiency Status - {deficiency_status}, SDI Value - {sdi_avg:.2f}. Provide village-specific fertilizer and crop recommendations."
    ai_insights = generate_ai_insights(prompt)

    # Display results
    print("\nSoil Deficiency Analysis:")
    print(f"Village: {village}")
    print(f"Predicted Deficiency Status: {deficiency_status}")
    print(f"SDI Value: {sdi_avg:.2f}")
    print(f"Recommendation: {resolution}")
    print("\nAI-Generated Insights:")
    print(ai_insights)
