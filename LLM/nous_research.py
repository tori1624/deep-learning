# app.py
import streamlit as st
import requests


# --- 기본 설정 ---
st.set_page_config(page_title="Hermes-3 Chat", layout="centered")
st.title("💬 Hermes-3 LLaMA 대화형 챗봇")

# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 옵션 조절 슬라이더 ---
st.sidebar.header("🛠️ 모델 옵션")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
max_tokens = st.sidebar.slider("Max Tokens", 50, 1024, 200, 10)

# --- 사용자 입력 받기 ---
user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    # 대화 기록에 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Hermes-3 API 요청
    url = "https://inference-api.nousresearch.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {api-key}"
    }
    data = {
        "model": "Hermes-3-Llama-3.1-70B",
        "messages": st.session_state.messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }

    with st.spinner("모델이 답변을 생성 중입니다..."):
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            reply = response.json()['choices'][0]['message']['content']
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            st.error("API 요청 실패: " + response.text)

# --- 대화 출력 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
