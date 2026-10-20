from transformers import AutoTokenizer, AutoModelForCausalLM

name = "LiquidAI/LFM2-350M-Extract"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")

messages = [
    {"role": "system", "content": 'Return data as a JSON object with the following schema:\n{"name": "string", "company": "string", "amount": "string"}'},
    {"role": "user", "content": "Hi, this is Karim from Atlas Corp. Please find invoice #123 for 4,500 MAD."},
]

inputs = tok.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

out = model.generate(**inputs, max_new_tokens=200, do_sample=False)  # greedy = temp 0
print(tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))