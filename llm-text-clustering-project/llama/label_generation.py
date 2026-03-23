import random
from openai import OpenAI
import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import time
import re

def ini_client():
    # Modification pour pointer vers ton serveur Ollama local
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    return client

def chat(prompt, client):
    # Utilisation du modèle llama3.1:8b téléchargé précédemment
    completion = client.chat.completions.create(
        model="llama3.1:8b",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON. You must return only a valid JSON object."},
            {"role": "user", "content": prompt}
        ]
    )
    response_origin = completion.choices[0].message.content
    return response_origin

def load_dataset(data_path, data, use_large):
    data_file = os.path.join(data_path, data, "large.jsonl") if use_large else os.path.join(data_path, data, "small.jsonl")
    print(f"Use dataset {data_file}")
    with open(data_file,'r') as f:
        data_list = []
        for line in f:
            json_object = json.loads(line)
            data_list.append(json_object)
    print(f"Length of dataset: {len(data_list)}")
    return data_list

def get_label_list(data_list):
    label_list = []
    for data in data_list:
        if data["label"] not in label_list:
            label_list.append(data["label"])
    return label_list

def prompt_construct_generate_label(sentence_list, given_labels):
    json_example = {"labels": ["label name", "label name"]}
    # Prompt optimisé pour Llama 3.1 pour éviter les hallucinations
    prompt = f"Given the labels, under a text classification scenario, can all these text match the label given? If the sentence does not match any of the label, please generate a meaningful new label name.\n" \
             f"Labels: {given_labels}\n" \
             f"Sentences: {sentence_list}\n" \
             f"IMPORTANT: You should NOT return meaningless label names such as 'new_label_1' or 'unknown_topic_1'. Return only the NEW label names in a JSON format like: {json_example}"
    return prompt

def prompt_construct_merge_label(label_list):
    json_example = {"merged_labels": ["label name", "label name"]}
    # Prompt renforcé pour forcer une fusion plus agressive (ton problème de trop de clusters)
    prompt = f"Analyze the following list of {len(label_list)} text labels. Identify synonyms and duplicate concepts. " \
             f"Your task is to merge similar entries into a single representative label to simplify the list and reduce redundancy. " \
             f"Do not create hierarchies. \n" \
             f"Labels: {label_list}.\n" \
             f"Return the final simplified list in JSON format like: {json_example}"
    return prompt

def get_sentences(sentence_list):
    return [i['input'] for i in sentence_list]

def label_generation(args, client, data_list, chunk_size):
    count = 0
    all_labels = []
    with open(args.given_label_path, 'r') as f:
        given_labels = json.load(f)
    
    for label in given_labels[args.data]:
        all_labels.append(label)
    
    # On parcourt par chunks
    for i in range(0, len(data_list), chunk_size):
        sentence_list = data_list[i:i+chunk_size]
        sentences = get_sentences(sentence_list)
        prompt = prompt_construct_generate_label(sentences, given_labels[args.data])
        
        origin_response = chat(prompt, client)
        if origin_response is None:
            continue
        
        count += 1
        try:
            # Sécurisation du JSON (Llama peut parfois ajouter du texte autour du bloc JSON)
            clean_json = re.search(r'\{.*\}', origin_response, re.DOTALL).group()
            response = json.loads(clean_json)
            
            key = list(response.keys())[0]
            if isinstance(response[key], list):
                current_labels = response[key]
                for label in current_labels:
                    # Nettoyage des labels inutiles
                    if any(x in label.lower() for x in ["unknown", "new_label", "other"]):
                        continue
                    if label not in all_labels:
                        all_labels.append(label)
            else:
                all_labels.append(response[key])
        except Exception as e:
            continue
        
        if args.print_details:
            print(f"Batch {count}: {len(all_labels)} total labels")
            
        if count >= args.test_num:
            break
            
    return all_labels

def merge_labels(args, all_labels, client): 
    if len(all_labels) < 2: return all_labels
    prompt = prompt_construct_merge_label(all_labels)
    response_raw = chat(prompt, client)
    try:
        clean_json = re.search(r'\{.*\}', response_raw, re.DOTALL).group()
        response = json.loads(clean_json)
        merged_labels = []
        for key, sub_label_list in response.items():
            if isinstance(sub_label_list, list):
                for label in sub_label_list:
                    merged_labels.append(label)
            else:
                merged_labels.append(sub_label_list)
        return list(set(merged_labels)) # Suppression des doublons exacts
    except:
        return all_labels

def write_dict_to_json(args, input, output_path, output_name):
    size = "large" if args.use_large else "small"
    file_name = os.path.join(output_path, '_'.join([args.data, size, output_name]) + ".json")
    with open(file_name, 'w') as json_file:
        json.dump(input, json_file, indent=2)
    print(f"JSON file '{file_name}' written.")

def main(args): 
    print(f"--- Starting Label Generation for {args.data} ---")
    print("use_large: ", args.use_large)
    start_time = time.time()
    client = ini_client()
    
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    random.shuffle(data_list)
    
    label_list = get_label_list(data_list) # Labels réels pour info
    print(f"Ground Truth cluster num: {len(label_list)}")
    write_dict_to_json(args, label_list, args.output_path, "true_labels")
    
    all_labels = label_generation(args, client, data_list, args.chunk_size)
    print(f"Total labels given by LLM (before merge): {len(all_labels)}")
    write_dict_to_json(args, all_labels, args.output_path, "llm_generated_labels_before_merge")
    
    print("Merging similar labels...")
    final_labels = merge_labels(args, all_labels, client)
    write_dict_to_json(args, final_labels, args.output_path, "llm_generated_labels_after_merge")
    
    print(f"Label number after merge: {len(final_labels)}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument("--data", type=str, default="arxiv_fine")
    parser.add_argument("--output_path", type=str, default="./generated_labels")
    parser.add_argument("--given_label_path", type=str, default="./generated_labels/chosen_labels.json")
    parser.add_argument("--use_large", action="store_true")
    parser.add_argument("--print_details", type=bool, default=True)
    parser.add_argument("--test_num", type=int, default=10) # Augmente ce chiffre pour plus de précision
    parser.add_argument("--chunk_size", type=int, default=15)
    args = parser.parse_args()
    main(args)