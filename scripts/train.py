"""
Fine-tune Qwen2.5-3B with QLoRA on chat data.
Target: RTX 3050 4GB VRAM.
"""
import json
from pathlib import Path
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq

# ---- Config ----
MODEL_NAME = "unsloth/Qwen2.5-3B-bnb-4bit"
MAX_SEQ_LEN = 1024
LORA_RANK = 32
LORA_ALPHA = 64

TRAIN_PATH = "data/formatted/train_split.jsonl"
VAL_PATH = "data/formatted/val.jsonl"
OUTPUT_DIR = "outputs/qwen25-3b-ch3sh1re"

# ---- Load model + tokenizer ----
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,           # auto: bf16 on Ampere
    load_in_4bit=True,
)

# ---- Apply LoRA adapters ----
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",  # ~30% less VRAM
    random_state=42,
)

# ---- Chat template (Qwen2.5) ----
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

def format_conversations(examples):
    texts = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

# ---- Load + format datasets ----
train_ds = load_dataset("json", data_files=TRAIN_PATH, split="train")
val_ds = load_dataset("json", data_files=VAL_PATH, split="train")

train_ds = train_ds.map(format_conversations, batched=True, remove_columns=train_ds.column_names)
val_ds = val_ds.map(format_conversations, batched=True, remove_columns=val_ds.column_names)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ---- Trainer ----
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=1,
        packing=False,

        # Memory-tight settings for 4GB
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,   # effective batch = 16
        gradient_checkpointing=True,

        # Schedule
        num_train_epochs=2,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,

        # Precision
        bf16=True,
        fp16=False,

        # Logging / saving
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=200,

        seed=42,
        report_to="none",
    ),
)

# ---- Train on assistant turns only ----
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# ---- Go ----
trainer.train()

# ---- Save LoRA adapter ----
model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print("Done.")
