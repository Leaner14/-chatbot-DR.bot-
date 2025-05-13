
import streamlit as st
from dotenv import load_dotenv
import os
import requests
import pandas as pd
from kagglehub import load_dataset, KaggleDatasetAdapter


# Try to load from secrets (Streamlit Cloud)
api_key = st.secrets.get("OPENROUTER_API_KEY", None)


# If not found in secrets.toml, try .env (for local dev)
if not api_key:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")

# Load environment variables
# load_dotenv()
# api_key = os.getenv("OPENROUTER_API_KEY")  # Your OpenRouter API key

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file.")

# Load medicine dataset from Kaggle
@st.cache_data
def load_medicine_data():
    return load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "drowsyng/medicines-dataset",
        "medicines.csv"
    )

df = load_medicine_data()

# Call the DeepSeek R1 model via OpenRouter

def get_deepseek_response(user_question):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": user_question}
        ]
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
        print("Status Code:", response.status_code)
        print("Raw Response:", response.text)

        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"❌ API Error: {e}"





# Get medicine info and AI advice
def get_medicine_and_advice(disease_input):
    matches = df[df['disease_name'].str.contains(disease_input, case=False, na=False)]
    reply = ""

    if not disease_input.strip():
        return "⚠️ Please enter a valid disease or symptom."

    if not matches.empty:
        reply += "🩺 **Medicines found in the dataset:**\n\n"
        for _, row in matches.iterrows():
            med_name = row['med_name']
            med_url = row['med_url']
            reply += f"- **{med_name}**: [View Details]({med_url})\n"
    else:
        reply += "⚠️ No medicines found in the dataset for that condition.\n"

    with st.spinner("💡 Getting AI treatment advice..."):
        ai_advice = get_deepseek_response(f"Give treatment advice for: {disease_input}")

    reply += f"\n\n💡 **AI Advice:**\n{ai_advice}"
    return reply

# Streamlit App UI
def main():
    st.set_page_config(page_title="DR.bot - Medical Assistant", page_icon="💊")
    st.title("💊 DR.bot")
    st.markdown("Ask me about any disease or symptom, and I'll suggest relevant medicines and AI-generated treatment advice!")

    disease_input = st.text_input("Enter disease or symptom:")

    if disease_input:
        response = get_medicine_and_advice(disease_input)
        st.markdown(response, unsafe_allow_html=True)

if __name__ == "__main__":
    main()