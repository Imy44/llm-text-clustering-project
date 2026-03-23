import random
import os
import json
import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ===============================
# INITIALISATION MODELE QWEN
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
        max_new_tokens=128,  # réduit pour accélérer
        temperature=0.3,
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

    with open(data_file, 'r') as f:
        for line in f:
            json_object = json.loads(line)
            data_list.append(json_object)

    print(f"Length of dataset: {len(data_list)}")

    return data_list


# ===============================
# EXTRACTION LABELS
# ===============================

def get_label_list(data_list):

    label_list = []

    for data in data_list:
        if data["label"] not in label_list:
            label_list.append(data["label"])

    return label_list


# ===============================
# PROMPT GENERATION LABEL
# ===============================

def prompt_construct_generate_label(sentence_list, given_labels):

    json_example = {"labels": ["label name", "label name"]}

    prompt = f"""
Given the labels in a text classification scenario, determine whether the following sentences match one of the existing labels.

Existing labels:
{given_labels}

Sentences:
{sentence_list}

If a sentence does not match any label, generate a meaningful new label.

Do NOT generate meaningless labels like new_label_1 or unknown_topic_1.

Return ONLY new labels in JSON format like:
{json_example}
"""

    return prompt


# ===============================
# PROMPT MERGE LABEL
# ===============================

def prompt_construct_merge_label(label_list):

    json_example = {"merged_labels": ["label name", "label name"]}

    prompt = f"""
Please analyze the provided list of labels to identify entries that are similar or duplicates.

Consider:
- synonyms
- variations in phrasing
- closely related concepts

Label list:
{label_list}

Return a simplified list in JSON format like:
{json_example}
"""

    return prompt


# ===============================
# EXTRAIRE PHRASES
# ===============================

def get_sentences(sentence_list):

    sentences = []

    for i in sentence_list:
        sentences.append(i["input"])

    return sentences


# ===============================
# GENERATION DES LABELS
# ===============================

def label_generation(args, tokenizer, model, data_list, chunk_size):

    all_labels = []

    with open(args.given_label_path, "r") as f:
        given_labels = json.load(f)

    for label in given_labels[args.data]:
        all_labels.append(label)

    total_batches = (len(data_list) + chunk_size - 1) // chunk_size

    for i in tqdm(
        range(0, len(data_list), chunk_size),
        total=total_batches,
        desc=f"{args.data} - Generating labels"
    ):

        sentence_list = data_list[i:i + chunk_size]

        sentences = get_sentences(sentence_list)

        prompt = prompt_construct_generate_label(
            sentences,
            given_labels[args.data]
        )

        response = chat(prompt, tokenizer, model)

        try:

            json_start = response.find("{")
            json_text = response[json_start:]

            response_dict = json.loads(json_text)

            current_labels = list(response_dict.values())[0]

            if isinstance(current_labels, list):

                for label in current_labels:

                    if "unknown_topic" in label or "new_label" in label:
                        continue

                    if label not in all_labels:
                        all_labels.append(label)

        except:
            continue

    return all_labels


# ===============================
# MERGE LABELS
# ===============================

def merge_labels(args, tokenizer, model, all_labels):

    prompt = prompt_construct_merge_label(all_labels)

    response = chat(prompt, tokenizer, model)

    try:

        json_start = response.find("{")
        json_text = response[json_start:]

        response_dict = json.loads(json_text)

        merged_labels = []

        for key, sub_label_list in response_dict.items():

            for label in sub_label_list:
                merged_labels.append(label)

        return merged_labels

    except:

        return all_labels


# ===============================
# WRITE JSON
# ===============================

def write_dict_to_json(args, data_input, output_path, output_name):

    size = "large" if args.use_large else "small"

    file_name = os.path.join(
        output_path,
        "_".join([args.data, size, output_name]) + ".json"
    )

    with open(file_name, "w") as json_file:
        json.dump(data_input, json_file, indent=2)

    print(f"JSON file '{file_name}' written.")


# ===============================
# MAIN
# ===============================

def main(args):

    start_time = time.time()

    # Chargement du modèle UNE SEULE FOIS
    tokenizer, model = init_model()

    data_list = load_dataset(
        args.data_path,
        args.data,
        args.use_large
    )

    random.shuffle(data_list)

    label_list = get_label_list(data_list)

    print(f"Total cluster num: {len(label_list)}")

    write_dict_to_json(
        args,
        label_list,
        args.output_path,
        "true_labels"
    )

    print(sorted(label_list))

    all_labels = label_generation(
        args,
        tokenizer,
        model,
        data_list,
        args.chunk_size
    )

    print(f"Total labels generated: {len(all_labels)}")

    write_dict_to_json(
        args,
        all_labels,
        args.output_path,
        "llm_generated_labels_before_merge"
    )

    final_labels = merge_labels(
        args,
        tokenizer,
        model,
        all_labels
    )

    write_dict_to_json(
        args,
        final_labels,
        args.output_path,
        "llm_generated_labels_after_merge"
    )

    print(f"Label number after merge: {len(final_labels)}")

    print(f"Total time usage: {time.time() - start_time} seconds")


# ===============================
# ARGUMENTS
# ===============================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument("--data", type=str, default="arxiv_fine")
    parser.add_argument("--output_path", type=str, default="./generated_labels")
    parser.add_argument("--given_label_path", type=str, default="./generated_labels/chosen_labels.json")

    parser.add_argument("--use_large", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=15)

    args = parser.parse_args()

    main(args)