# 🇮🇳 Indian Legal Migration Assistant ⚖️

An AI-powered legal assistant that explains **Indian Penal Code (IPC)** sections
and maps them to **Bharatiya Nyaya Sanhita (BNS)** equivalents.

This project fine-tunes **TinyLlama-1.1B-Chat** using **LoRA** on Indian legal data
and provides responses through a **Streamlit chatbot interface**.

---

## 🚀 Features

- Explain IPC sections
- Map IPC to BNS laws
- Fine-tuned LLM using LoRA
- Streamlit interactive chatbot
- HuggingFace hosted model
- GPU training with RTX 3050

---

## 🧠 Model

Base Model  
TinyLlama-1.1B-Chat

Fine-tuning  
LoRA (PEFT)

Model on HuggingFace  
https://huggingface.co/Yashdew/ipc-bns-legal-assistant

---

## ⚙️ Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
