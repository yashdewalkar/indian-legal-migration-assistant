from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "models/ipc_bns_model"

# load base model
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

# load adapter
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# repository name
repo_id = "Yashdew/ipc-bns-legal-assistant"

# push
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)

print("Model uploaded successfully!")