import json
from llm_sdk.llm_sdk import Small_LLM_Model


active = True
if active:
    model = Small_LLM_Model()



definitions = [
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {
        "a": {
            "type": "number"
        },
        "b": {
            "type": "number"
        }
        },
        "returns": {
        "type": "number"
        }
    },
    {
        "name": "fn_greet",
        "description": "Generate a greeting message for a person by name.",
        "parameters": {
        "name": {
            "type": "string"
        }
        },
        "returns": {
        "type": "string"
        }
    },
    {
        "name": "fn_reverse_string",
        "description": "Reverse a string and return the reversed result.",
        "parameters": {
        "s": {
            "type": "string"
        }
        },
        "returns": {
        "type": "string"
        }
    }
]



func_names_list = []
func_parameters = {}

def create_data_works():
    for d in definitions:
        func_names_list.append(d["name"])
        temp = {}
        for k, v in d["parameters"].items():
            temp.update({k: v["type"]})
        func_parameters[d["name"]] = temp
create_data_works()




prompt = "What is the sum of 2 and 3?"

real_prompt = ""
for d in definitions:
    real_prompt += "\n"
    real_prompt += d["name"]
    real_prompt += ": "
    real_prompt += d["description"]
real_prompt += "\n\nJSON:"





ids_of_prompt_json = model.encode(real_prompt)[0].tolist()
ids_of_json = []



state = "OPEN"

while True:

    


