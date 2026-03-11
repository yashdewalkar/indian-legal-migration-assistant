from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/final_sft/train.jsonl",
        "validation": "data/final_sft/val.jsonl"
    }
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---- 4-bit quantization (important for your lapto ----
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda"
)

print("Applying LoRA...")

peft_config = LoraConfig(
    r=8,                # smaller rank for faster training
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

training_config = SFTConfig(
    output_dir="models/ipc_bns_model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=5,
    save_strategy="epoch",
    max_length=256,
    dataset_text_field="text",
    fp16=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=training_config,
    peft_config=peft_config,
)

print("Starting training...")

trainer.train()

trainer.save_model("models/ipc_bns_model")

print("Training completed!")