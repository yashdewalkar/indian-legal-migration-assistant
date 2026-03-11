from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "models/ipc_bns_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda"
)

prompt = """
### Instruction:
Explain IPC Section 302

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=120
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))