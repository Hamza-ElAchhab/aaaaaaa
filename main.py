from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto")





# prompt = "explain to ne what is AI?"

# inputs = tokenizer(prompt, return_tensors="pt")

# outputs = model.generate(inputs["input_ids"], max_new_tokens=50)


# print("\n\n\n")
# print(tokenizer.decode(outputs[0]))





context = """
Artificial Intelligence (AI) is a field of computer science that focuses on building systems
capable of performing tasks that normally require human intelligence, such as learning,
reasoning, and problem-solving. AI is used in education, healthcare, and industry.
"""



question = "What is AI used for?"

prompt = f"""
Context:
{context}
Question:
{question}
Answer ONLY using the context:
"""




inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.2
)


print()
print()
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
