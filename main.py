import json
from llm_sdk.llm_sdk import Small_LLM_Model
import json
import numpy



model = Small_LLM_Model()



def load_voca():
    with open(model.get_path_to_vocab_file(), "r") as f:
        data_str = f.read()
        str_id = json.loads(data_str)
    id_str = {}
    for s, i in str_id.items():
        id_str[i] = s
    return str_id, id_str


str_to_id ,id_to_str = load_voca()



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



def masking(logits, alloweds=None):

    arr_logits = numpy.array(logits, numpy.float64)

    mask = numpy.full_like(arr_logits, -numpy.inf)

    if alloweds == None:
        mask = numpy.full_like(arr_logits, 0.0)
    else:
        mask[list(alloweds)] = 0.0

    return int(numpy.argmax((arr_logits + mask)))






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
current_func_name = ""



while True:
    
    if state == "OPEN":
        ids_of_json.append(str_to_id["{"])
        ids_of_prompt_json.append(str_to_id["{"])
        state = "NAME"
        continue


    if state == "NAME":
        lst = model.encode('"name"')[0].tolist()
        for ele in lst:
            ids_of_prompt_json.append(ele)
            ids_of_json.append(ele)
        state = "COLON1"
        continue


    if state == "COLON1":
        ids_of_json.append(str_to_id[":"])
        ids_of_prompt_json.append(str_to_id[":"])
        state = "FUN_NAME"
        continue


    if state == "FUN_NAME":
        buffer = ""
        functions_encoded = []
        for n in func_names_list:
            functions_encoded.append(model.encode(f'"{n}"')[0].tolist())

        pos = 0
        while True:
            alloweds = set()

            for lst in functions_encoded:
                alloweds.add(lst[pos])


            logits = model.get_logits_from_input_ids(ids_of_prompt_json)
            next_id = masking(logits, alloweds)

            ids_of_json.append(next_id)
            ids_of_prompt_json.append(next_id)
            buffer += (model.decode(next_id))
            
            functions_encoded = [
                lst for lst in functions_encoded if len(lst) > pos and lst[pos] == next_id
            ]

            pos += 1

            if len(functions_encoded) == 1 and pos == len(functions_encoded[0]):
                break

        current_func_name = buffer[1:-1]
        state = "COMMA"
        continue


    if state == "COMMA":
        ids_of_json.append(str_to_id[","])
        ids_of_prompt_json.append(str_to_id[","])
        state = "PARAMETERS"
        continue


    if state == "PARAMETERS":
        lst = model.encode('"parameters"')[0].tolist()
        for ele in lst:
            ids_of_prompt_json.append(ele)
            ids_of_json.append(ele)
        state = "COLON2"
        continue


    if state == "COLON2":
        ids_of_json.append(str_to_id[":"])
        ids_of_prompt_json.append(str_to_id[":"])
        state = "OPEN_PARAMETER"
        continue


    if state == "OPEN_PARAMETER":
        ids_of_json.append(str_to_id["{"])
        ids_of_prompt_json.append(str_to_id["{"])
        state = "KEYS_PARAMETER"
        continue


    if state == "KEYS_PARAMETER":
        break










print(model.decode(ids_of_json))

print(current_func_name)
