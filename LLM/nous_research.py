import streamlit as st
import requests

# --- Basic Config ---
st.set_page_config(page_title="Nous Research Chat", layout="centered")
st.title("💬 Nous Research Chatbot")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False

# --- Hermes API Info ---
API_URL = "https://inference-api.nousresearch.com/v1/chat/completions"

# --- Model Options ---
MODEL_OPTIONS = {
    "Hermes-3-Llama-3.1-70B": "Hermes-3-Llama-3.1-70B",
    "DeepHermes-3-Llama-3-8B-Preview": "DeepHermes-3-Llama-3-8B-Preview",
    "DeepHermes-3-Mistral-24B-Preview": "DeepHermes-3-Mistral-24B-Preview",
    "Hermes-3-Llama-3.1-405B": "Hermes-3-Llama-3.1-405B"
}

# --- Sidebar: API Key ---
st.sidebar.header("🔑 API Key Configuration")
user_api_key = st.sidebar.text_input("Enter your API Key", type="password")

# --- Validate API Key ---
def validate_api_key(api_key: str) -> bool:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    test_data = {
        "model": "Hermes-3-Llama-3.1-70B",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1
    }
    try:
        response = requests.post(API_URL, headers=headers, json=test_data, timeout=5)
        return response.status_code == 200
    except:
        return False

if user_api_key:
    if user_api_key != st.session_state.api_key:
        if validate_api_key(user_api_key):
            st.session_state.api_key = user_api_key
            st.session_state.api_key_valid = True
            st.sidebar.success("✅ API key is valid and registered!")
        else:
            st.session_state.api_key_valid = False
            st.sidebar.error("❌ Invalid API key. Please try again.")

# --- Sidebar: Model Selection and Parameters ---
st.sidebar.header("🧠 Model Selection")
selected_model_label = st.sidebar.selectbox("Choose a model", list(MODEL_OPTIONS.keys()))
selected_model = MODEL_OPTIONS[selected_model_label]

st.sidebar.header("🛠️ Model Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
top_p = st.sidebar.slider("Top-p (Nucleus Sampling)", 0.0, 1.0, 0.9, 0.05)
max_tokens = st.sidebar.slider("Max Tokens", 50, 1024, 200, 10)

# --- Chat Input (only if valid key) ---
if st.session_state.api_key_valid:
    user_input = st.chat_input("Type your message...")
else:
    st.info("Please enter a valid API key in the sidebar to start chatting.")
    user_input = None

# --- Handle User Input ---
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {st.session_state.api_key}"
    }
    data = {
        "model": selected_model,
        "messages": st.session_state.messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }

    with st.spinner("Generating response..."):
        response = requests.post(API_URL, headers=headers, json=data)

        if response.status_code == 200:
            reply = response.json()['choices'][0]['message']['content']
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            st.error("API request failed: " + response.text)

# --- Display Chat Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
