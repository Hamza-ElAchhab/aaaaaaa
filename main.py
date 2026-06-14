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
- Ali is a student at a computer science school.
He is working on a project about Retrieval-Augmented Generation (RAG).
His project uses Qwen model and BM25 for searching code snippets.
Ali lives in Morocco.
His favorite programming language is Python.
"""
question = "What project is Ali working on and which model does he use?"




fully_prompt = f"""
You are a strict question answering system for the vLLM codebase.

Rules:
- Use ONLY the provided context.
- Do NOT use any external knowledge.
- Do NOT explain your answer.
- Return ONLY one short answer sentence.
- If the answer is not in the context, reply exactly:
I could not find the answer in the retrieved sources.

Context:
{context}

Question:
{question}

Answer:

"""



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
        max_new_tokens=100,
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

