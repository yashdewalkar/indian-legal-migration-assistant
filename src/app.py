import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# -------- CONFIG --------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "models/ipc_bns_model"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------- LOAD MODEL --------
@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    model = model.to(DEVICE)
    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


# -------- PAGE CONFIG --------
st.set_page_config(
    page_title="Indian Legal AI",
    page_icon="⚖️",
    layout="wide"
)

st.title("🇮🇳 Indian Legal Migration Assistant ⚖️")
st.caption("Ask questions about IPC sections and their BNS equivalents")


# -------- SESSION STATE --------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -------- DISPLAY CHAT HISTORY --------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------- USER INPUT --------
if prompt := st.chat_input("Ask a legal question..."):

    # show user message
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # create prompt for model
    model_prompt = f"""
### Instruction:
{prompt}

### Response:
"""

    inputs = tokenizer(model_prompt, return_tensors="pt").to(DEVICE)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing law..."):

            with torch.no_grad():

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.2,
                    top_p=0.9,
                    do_sample=True
                )

            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "### Response:" in answer:
                answer = answer.split("### Response:")[-1].strip()

            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })