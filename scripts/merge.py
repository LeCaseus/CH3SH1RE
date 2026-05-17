from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/qwen25-3b-ch3sh1re/final",
    max_seq_length=1024,
    load_in_4bit=False,
    device_map="cpu",
)

model.save_pretrained_merged(
    "outputs/qwen25-3b-ch3sh1re/merged-fp16",
    tokenizer,
    save_method="merged_16bit",
)
