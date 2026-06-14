from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


model_name = "Qwen/Qwen3-0.6B"

#install the tokenizer for the type of model name
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-0.6B",
    trust_remote_code=True
)


#install the model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=torch.float16,
    device_map="auto"
)





context = """
the small baby's name is thrthng.
"""
question = "What is the name of small baby?"



# prompt = f"""
# Context:
# {context}

# Question:
# {question}

# Instruction:
# - Answer ONLY from the context.
# - Do NOT add extra information.
# - Give a short direct answer only.
# """



model.eval()



# use obj as fun cause this -> : __call__
inputs = tokenizer(
    question,
    return_tensors="pt",
    truncation=True,
    max_length=4096,
    padding=False,
    return_attention_mask=True
)



print(inputs)


# outputs = model.generate(
#     **tokens,
#     max_new_tokens=50,
#     temperature=0.2
# )



# print()
# print()
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))







# prompt = f"""
# You are answering questions about the vLLM codebase.

# Use ONLY the provided context.

# If the answer is not present in the context,
# reply exactly:

# I could not find the answer in the retrieved sources.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """