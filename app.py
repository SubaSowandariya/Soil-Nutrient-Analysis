from flask import Flask, render_template, request, jsonify
from deep_translator import GoogleTranslator
import pandas as pd
import joblib
import numpy as np
import os
import catboost as cb
from dotenv import load_dotenv
from groq import Groq

# ✅ Load environment variables
load_dotenv()

# ✅ Set up Groq API for AI insights
api_key = os.getenv("GROQ_API_KEY")
if api_key is None:
    raise ValueError("API key not found. Please set the GROQ_API_KEY environment variable.")
client = Groq(api_key=api_key)

# ✅ Load all datasets
macro_df = pd.read_csv("Processed_Macronutrients.csv")
micro_df = pd.read_csv("Processed_Micronutrients.csv")
cleaned_macro_df = pd.read_csv("Cleaned_Macronutrients.csv")
cleaned_micro_df = pd.read_csv("Cleaned_Micronutrients.csv")
irrigation_data = pd.read_csv("Updated_Irrigation_Data.csv")

# ✅ Merge soil datasets
df_soil = pd.merge(macro_df, micro_df, on="Village")
df_cleaned_soil = pd.merge(cleaned_macro_df, cleaned_micro_df, on="village")

# ✅ Extract unique village names and save to CSV
all_villages = pd.concat([df_soil["Village"], irrigation_data["village"]]).drop_duplicates().str.lower().sort_values()
unique_villages_df = pd.DataFrame(all_villages.unique(), columns=["village"])
unique_villages_df.to_csv("unique_villages.csv", index=False)

# ✅ Load pre-trained models
scaler = joblib.load("scaler_model.pkl")
kmeans = joblib.load("kmeans_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")

# ✅ Load trained CatBoost model
catboost_model = cb.CatBoostClassifier()
catboost_model.load_model("catboost_model.cbm")

# ✅ Fertilizer recommendations
fertilizer_map = {
    "Nitrogen - Low": ("Urea or Ammonium Nitrate", "Legumes, Corn"),
    "Phosphorous - Low": ("Single Super Phosphate (SSP) or DAP", "Wheat, Barley"),
    "Potassium - Low": ("Muriate of Potash (MOP) or Potassium Sulfate", "Banana, Potatoes"),
    "OC - Low": ("Compost or Organic Manure", "Vegetables, Fruits"),
    "EC - Saline": ("Apply Gypsum or Improve Drainage", "Salt-tolerant crops like Barley"),
    "pH - Acidic": ("Use Lime (Calcium Carbonate)", "Legumes, Alfalfa"),
    "pH - Alkaline": ("Use Sulfur-based fertilizers", "Corn, Cotton"),
    "Copper - Deficient": ("Use Copper Sulfate", "Wheat, Citrus"),
    "Boron - Deficient": ("Use Borax or Boric Acid", "Carrots, Sunflower"),
    "S - Deficient": ("Use Gypsum or Ammonium Sulfate", "Mustard, Onion"),
    "Fe - Deficient": ("Use Ferrous Sulfate", "Spinach, Rice"),
    "Zn - Deficient": ("Use Zinc Sulfate", "Corn, Grapes"),
    "Mn - Deficient": ("Use Manganese Sulfate", "Soybeans, Pineapple")
}

# ✅ AI Insights Function
def generate_ai_insights(context):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": context}],
        model="llama-3.3-70b-versatile"
    )
    return response.choices[0].message.content

# ✅ Function to get irrigation plan
def irrigation_plan(village_name):
    village_name_lower = village_name.lower()
    irrigation_data["village_lower"] = irrigation_data["village"].str.lower()

    if village_name_lower not in irrigation_data["village_lower"].values:
        return "<p><strong>Village Not Found</strong></p>"

    village_data = irrigation_data[irrigation_data["village_lower"] == village_name_lower].iloc[0]
    wri = village_data["Predicted_Water_Requirement"]

    if wri < irrigation_data["WRI"].quantile(0.33):
        schedule = "Low Water Requirement: Use drip irrigation"
    elif wri < irrigation_data["WRI"].quantile(0.66):
        schedule = "Medium Water Requirement: Irrigate every 3 days"
    else:
        schedule = "High Water Requirement: Use rainwater harvesting & sprinklers"

    ai_prompt = f"What are the best irrigation practices for {village_name}?"
    ai_insights = generate_ai_insights(ai_prompt)

    return f"""
        <h3>Irrigation Plan for {village_name}</h3>
        <p><strong>Irrigation Schedule:</strong> {schedule}</p>
        <h4>AI Insights:</h4>
        <p>{ai_insights}</p>
    """
def soil_deficiency_analysis(village_name):
    village_name_lower = village_name.lower()  # Convert input to lowercase
    df_soil["village_lower"] = df_soil["Village"].str.lower()  # Add lowercase column

    if village_name_lower not in df_soil["village_lower"].values:
        return "<p><strong>Village Not Found</strong></p>"

    village_index = df_soil[df_soil["village_lower"] == village_name_lower].index.tolist()
    village_features = df_soil.loc[village_index, df_soil.columns[1:-1]].values
    prediction = catboost_model.predict(village_features)[0]

    deficiency_status = "Deficient" if prediction == 1 else "Not Deficient"

    deficient_nutrients = []
    for nutrient, (fertilizer, crops) in fertilizer_map.items():
        if nutrient in df_soil.columns and df_soil.loc[village_index, nutrient].values[0] == 1:
            deficient_nutrients.append(f"<li><strong>{nutrient}:</strong> {fertilizer} (Recommended Crops: {crops})</li>")

    if prediction == 1 and not deficient_nutrients:
        deficiency_status = "Not Deficient"

    if deficiency_status == "Deficient":
        recommendation = "<ul>" + "".join(deficient_nutrients) + "</ul>"
        ai_prompt = f"Soil analysis for {village_name}: The soil is Deficient. Deficiencies include {', '.join([nutrient.split(':')[0] for nutrient in deficient_nutrients])}. Provide best practices for soil improvement."
    else:
        recommendation = "<p>No deficiencies found. Maintain the current agricultural practices.</p>"
        ai_prompt = f"Soil analysis for {village_name}: The soil is Not Deficient. Provide general best practices for maintaining soil health."

    ai_insights = generate_ai_insights(ai_prompt)

    return f"""
    <h2>Soil Deficiency Analysis for {village_name}</h2>
    <p><strong>Deficiency Status:</strong> {deficiency_status}</p>
    <h3>Recommended Fertilizers:</h3>
    {recommendation}
    <h3>AI Insights:</h3>
    <p>{ai_insights}</p>
    """
# ✅ Flask App
app = Flask(__name__)

@app.route('/get_villages', methods=['GET'])
def get_villages():
    """Returns a list of village names matching user input dynamically."""
    user_input = request.args.get("query", "").lower()

    if not user_input:
        return jsonify([])

    # Load unique village names
    villages_df = pd.read_csv("unique_villages.csv")
    villages_list = villages_df["village"].str.lower().tolist()

    # Filter by startswith or contains
    matches = [village for village in villages_list if village.startswith(user_input) or user_input in village]

    return jsonify(matches)

@app.route('/')
def about_first():
    return render_template('about.html')

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    village_name = request.form['village']
    analysis_type = request.form['analysis_type']

    response = f"<h2>Analysis Report for {village_name}</h2>"

    if analysis_type == "irrigation":
        response += irrigation_plan(village_name)
    elif analysis_type == "soil_deficiency":
        response += soil_deficiency_analysis(village_name)
    else:  # Both
        response += irrigation_plan(village_name) + "<br><br>" + soil_deficiency_analysis(village_name)

    return render_template('index.html', village=village_name, result=response)

def split_text(text, max_length=5000):
    """Splits text into chunks within the character limit without breaking words."""
    chunks = []
    while len(text) > max_length:
        split_index = text[:max_length].rfind(" ")  # Find last space within limit
        if split_index == -1:
            split_index = max_length  # If no space found, force a split
        chunks.append(text[:split_index])
        text = text[split_index:].lstrip()
    chunks.append(text)  # Add remaining text
    return chunks

@app.route('/translate', methods=['POST'])
def translate():
    try:
        text = request.form.get('text', '')
        if not text:
            return render_template('index.html', result="No text provided.")

        # Split long text while keeping words intact
        text_chunks = split_text(text)

        # Translate each chunk separately
        translated_chunks = [GoogleTranslator(source="en", target="ta").translate(chunk) for chunk in text_chunks]

        # Combine translated chunks back into full text
        translated_text = " ".join(translated_chunks)

        return render_template('index.html', result=translated_text)
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
