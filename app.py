import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
import json 
import joblib

# === LOAD CSS ===
def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# === KONFIGURASI ===
API_KEY = "AIzaSyD__OOKMY8wXu1Sudpcyh3ECdBYlj09h2U"
csv_path = r"data.csv"
history_path = "chat_history.json"
# rf_model_path = "model_rf_tfidf.joblib" 
# vectorizer_path = "tfidf_vectorizer.joblib" 

# === LOAD RANDOM FOREST ===
# if os.path.exists(rf_model_path) and os.path.exists(vectorizer_path):
#     rf_model = joblib.load(rf_model_path)
#     vectorizer = joblib.load(vectorizer_path)
# else:
#     rf_model, vectorizer = None, None
#     st.warning("Model atau vectorizer Random Forest tidak ditemukan!")

# def predict_rf(text):
#     if rf_model and vectorizer:
#         x_input = vectorizer.transform([text])
#         pred = rf_model.predict(x_input)
#         return pred[0]
#     return "Model RF belum siap."

# === FUNGSI GEMINI ===
def chat_gemini(contexts, history, question):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        api_key=API_KEY
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use relevant information from data and chat history."),
        ("human", "Data: {contexts}\n\nChat history: {history}\n\nUser question: {question}")
    ])
    try:
        chain = prompt_template | llm
        completion = chain.invoke({
            "contexts": contexts[:5000],
            "history": history[-3000:],
            "question": question
        })
        meta = getattr(completion, "response_metadata", {})
        usage = meta.get("usage_metadata", {})
        return completion.content, usage.get("prompt_token_count", 0), usage.get("candidates_token_count", 0)
    except Exception as e:
        if "quota" in str(e).lower():
            return "⚠️ Kuota Gemini API habis.", 0, 0
        return f"Error: {e}", 0, 0

# === SIMPAN DAN LOAD HISTORY ===
def save_history():
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "all_chats": st.session_state.all_chats,
            "active_chat": st.session_state.active_chat
        }, f, ensure_ascii=False, indent=2)

def load_history():
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                st.session_state.all_chats = data.get("all_chats", {"Chat 1": []})
                st.session_state.active_chat = data.get("active_chat", "Chat 1")
            else:
                st.session_state.all_chats = {"Chat 1": data}
                st.session_state.active_chat = "Chat 1"
    else:
        st.session_state.all_chats = {"Chat 1": []}
        st.session_state.active_chat = "Chat 1"

# === STATE CHAT ===
if "all_chats" not in st.session_state:
    load_history()
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {"Chat 1": []}
if "active_chat" not in st.session_state:
    st.session_state.active_chat = "Chat 1"

# === SIDEBAR ===
st.sidebar.markdown("## Riwayat Chat")

# Tombol buat chat baru (pastikan hanya satu kali dipanggil)
if st.sidebar.button(" New Chat", key="new_chat_button"):
    new_id = f"Chat {len(st.session_state.all_chats) + 1}"
    st.session_state.all_chats[new_id] = []
    st.session_state.active_chat = new_id
    save_history()
    st.rerun()

# Tampilkan daftar chat sebagai tombol list
for i, chat_id in enumerate(list(st.session_state.all_chats.keys())):
    is_active = chat_id == st.session_state.active_chat
    chat_label = f"🗨️ {chat_id}" if not is_active else f"👉 {chat_id}"

    col1, col2 = st.sidebar.columns([8, 1])
    with col1:
        if st.button(chat_label, key=f"select_{i}"):
            st.session_state.active_chat = chat_id
            save_history()
            st.rerun()
    with col2:
        if st.button("🗑️", key=f"delete_{i}"):
            del st.session_state.all_chats[chat_id]
            if not st.session_state.all_chats:
                st.session_state.all_chats["Chat 1"] = []
            st.session_state.active_chat = list(st.session_state.all_chats.keys())[0]
            save_history()
            st.rerun()

st.sidebar.caption("Klik nama chat di atas untuk berpindah.")

# === HEADER UTAMA ===
st.markdown("""
<div class="header">
    <h1>🤖 AI Chatbot Assistant</h1>
    <h3>Analisis Sentimen Tren <span style="color:#F97316;">#KaburAjaDulu</span></h3>
</div>
""", unsafe_allow_html=True)

# === LOAD DATA ===
contexts = ""
if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path, sep=";", on_bad_lines="skip", encoding="utf8").drop_duplicates()
        contexts = df[['tweet', 'label']].to_string(index=False)
    except Exception as e:
        st.error(f"Error membaca CSV: {e}")
else:
    st.warning(f"File tidak ditemukan di path: {csv_path}")

# === CHAT AREA ===
st.markdown("---")
st.markdown(f"### {st.session_state.active_chat}")

messages = st.session_state.all_chats[st.session_state.active_chat]

# Tampilkan pesan terakhir (maks 10)
for message in messages[-10:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input user (biarkan default style Streamlit)
if prompt := st.chat_input("Tanyakan sesuatu..."):
    history = "\n".join([f"{m['role']}: {m['content']}" for m in messages[-10:]])
    st.chat_message("user").markdown(prompt)
    messages.append({"role": "user", "content": prompt})

    with st.spinner("Menunggu respon dari Gemini..."):
        answer, in_toks, out_toks = chat_gemini(contexts, history, prompt)

    # Balasan dari AI
    with st.chat_message("assistant"):
        st.markdown(answer)

    messages.append({"role": "assistant", "content": answer})

    # Ganti nama chat otomatis dari pertanyaan pertama
    if st.session_state.active_chat.startswith("Chat") and len(messages) == 2:
        new_name = prompt[:25] + "..." if len(prompt) > 25 else prompt
        st.session_state.all_chats[new_name] = st.session_state.all_chats.pop(st.session_state.active_chat)
        st.session_state.active_chat = new_name

    st.session_state.all_chats[st.session_state.active_chat] = messages
    save_history()

    st.rerun()
