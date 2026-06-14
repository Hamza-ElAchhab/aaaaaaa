from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os



#install the tokenizer for the type of model name
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-0.6B",
    trust_remote_code=True
)

#install the model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)




context = """
the small baby's name is koko.
"""
question = "What is the name of small baby?"

fully_prompt = f"""
Context:
{context}

Question:
{question}

Instructions:
- Answer ONLY from the context.
- If the answer is not in the context, say:
  "I don't know based on the provided context."

Answer:
"""
print(f"You Give Him : {len(tokenizer.encode(fully_prompt))} token\n\n")





model.eval()
# use obj as fun cause this -> : __call__
inputs = tokenizer(
    fully_prompt,           #fully_prompt
    return_tensors="pt",
    truncation=True,
    max_length=5000     #Tokens Context Window, but more large more bad answers
)
#input will includes two keys tesor and attention rules by index


# with just for create them in memory and free up
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.0,                       #random
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,    #just for ignore some wornings
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )




#get only the answer without repeat prompts:
input_len = len(inputs["input_ids"][0].tolist())
new_tokens = outputs[0][input_len:]
answer = tokenizer.decode(new_tokens, skip_special_tokens=True)


print(answer.strip())















#in project :
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



# prompt = f"""
# Context:
# {context}

# Question:
# {question}

# Instructions:
# - Answer ONLY from the context.
# - If the answer is not in the context, say:
#   "I don't know based on the provided context."

# Answer:
# """