import os
import json
import argparse
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ===============================
# INITIALISATION MODELE
# ===============================

def init_model():

    model_name = "Qwen/Qwen3.5-9B"

    print("Loading Qwen3.5-9B model...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Model loaded successfully")

    return tokenizer, model


# ===============================
# GENERATION TEXTE
# ===============================

def chat(prompt, tokenizer, model):

    messages = [
        {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


# ===============================
# LOAD DATASET
# ===============================

def load_dataset(data_path, data, use_large):

    data_file = os.path.join(
        data_path,
        data,
        "large.jsonl" if use_large else "small.jsonl"
    )

    print(f"Use dataset {data_file}")

    data_list = []

    with open(data_file, "r") as f:
        for line in f:
            data_list.append(json.loads(line))

    print(f"Length of dataset: {len(data_list)}")

    return data_list


# ===============================
# LOAD PREDICT LABELS
# ===============================

def get_predict_labels(output_path, data):

    file_path = os.path.join(
        output_path,
        f"{data}_small_llm_generated_labels_after_merge.json"
    )

    with open(file_path, "r") as f:
        label_list = json.load(f)

    label_list = list(set(label_list))

    return label_list


# ===============================
# PROMPT CLASSIFICATION
# ===============================

def prompt_construct(label_list, sentence):

    json_example = {"label_name": "label"}

    prompt = f"""
Given the following label list and sentence, classify the sentence into the most appropriate label.

Label list:
{label_list}

Sentence:
{sentence}

Return ONLY the label name in JSON format like:
{json_example}
"""

    return prompt


# ===============================
# EXTRACTION LABEL
# ===============================

def answer_process(response, label_list):

    try:

        json_start = response.find("{")

        json_text = response[json_start:]

        response_dict = json.loads(json_text)

        predicted_label = list(response_dict.values())[0]

        if predicted_label in label_list:
            return predicted_label

    except:
        pass

    return "Unsuccessful"


# ===============================
# CLASSIFICATION
# ===============================

def known_label_categorize(args, tokenizer, model, data_list, label_list):

    answer = {}

    for label in label_list:
        answer[label] = []

    answer["Unsuccessful"] = []

    total_samples = len(data_list)

    for i in tqdm(range(total_samples), desc=f"{args.data} - Classification"):

        sentence = data_list[i]["input"]

        prompt = prompt_construct(label_list, sentence)

        response = chat(prompt, tokenizer, model)

        predicted_label = answer_process(response, label_list)

        if predicted_label in label_list:
            answer[predicted_label].append(sentence)
        else:
            answer["Unsuccessful"].append(sentence)

    return answer


# ===============================
# WRITE JSON
# ===============================

def write_answer_to_json(args, answer, output_path, output_name):

    size = "large" if args.use_large else "small"

    file_name = os.path.join(
        output_path,
        "_".join([args.data, size, output_name])
    )

    with open(file_name, "w") as json_file:
        json.dump(answer, json_file, indent=2)

    print(f"JSON file '{file_name}' written.")


# ===============================
# MAIN
# ===============================

def main(args):

    start_time = time.time()

    tokenizer, model = init_model()

    data_list = load_dataset(
        args.data_path,
        args.data,
        args.use_large
    )

    label_list = get_predict_labels(
        args.output_path,
        args.data
    )

    print(f"Number of labels used for classification: {len(label_list)}")

    answer = known_label_categorize(
        args,
        tokenizer,
        model,
        data_list,
        label_list
    )

    answer = {k: v for k, v in answer.items() if len(v) != 0}

    write_answer_to_json(
        args,
        answer,
        args.output_path,
        args.output_file_name
    )

    print("Classification finished")

    print(f"Total time usage: {time.time() - start_time} seconds")


# ===============================
# ARGUMENTS
# ===============================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument("--data", type=str, default="arxiv_fine")
    parser.add_argument("--output_path", type=str, default="./generated_labels")
    parser.add_argument("--output_file_name", type=str, default="find_labels.json")

    parser.add_argument("--use_large", action="store_true")

    args = parser.parse_args()

    main(args)