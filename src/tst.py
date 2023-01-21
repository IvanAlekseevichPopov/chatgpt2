from transformers import GPT2Tokenizer, GPT2LMHeadModel


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# generate text
prompt = "What is the meaning of life?"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
generated_text = model.generate(input_ids, max_length=2048)
generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
print(generated_text)