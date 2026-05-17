from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/qwen25-3b-ch3sh1re/final",
    max_seq_length=1024,
    load_in_4bit=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
FastLanguageModel.for_inference(model)

messages = [{"role": "user", "content": "hey, what's up?"}]

prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(repr(prompt))

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to("cuda")

out = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.1,
    do_sample=True,
    eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
    pad_token_id=tokenizer.pad_token_id,
)

print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False))
