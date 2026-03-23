import random
from openai import OpenAI
import os
import json
import argparse
import time
import re
from tqdm import tqdm

def ini_client():
    # Connexion à Ollama local
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    return client

def chat(prompt, client):
    try:
        completion = client.chat.completions.create(
            model="llama3.1:8b",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON. You must return only a valid JSON object."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return None

def load_dataset(data_path, data, use_large):
    suffix = "large.jsonl" if use_large else "small.jsonl"
    data_file = os.path.join(data_path, data, suffix)
    print(f"Loading dataset: {data_file}")
    data_list = []
    with open(data_file,'r') as f:
        for line in f:
            data_list.append(json.loads(line))
    return data_list

def get_predict_labels(output_path, data):
    # Charge les labels fusionnés à l'étape précédente
    data_file = os.path.join(output_path, f"{data}_small_llm_generated_labels_after_merge.json")
    with open(data_file, 'r') as f:
        return list(set(json.load(f)))

def prompt_construct(label_list, sentence):
    json_example = {"label_name": "chosen label"}
    prompt = f"Categorize the following sentence into ONE of the labels from the provided list.\n" \
             f"Label list: {label_list}\n" \
             f"Sentence: {sentence}\n" \
             f"Return only the chosen label name in JSON format like: {json_example}"
    return prompt

def answer_process(response, label_list):
    try:
        clean_json = re.search(r'\{.*\}', response, re.DOTALL).group()
        res_dict = json.loads(clean_json)
        val = list(res_dict.values())[0]
        # Vérification si le label retourné existe bien dans notre liste
        if val in label_list:
            return val
        # Si le LLM a un peu modifié le nom, on cherche le plus proche
        for l in label_list:
            if l.lower() in str(val).lower():
                return l
        return "Unsuccessful"
    except:
        return "Unsuccessful"

def main(args):
    client = ini_client()
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    label_list = get_predict_labels(args.output_path, args.data)
    
    print(f"Classifying {len(data_list)} samples for {args.data}...")
    results = {label: [] for label in label_list}
    results["Unsuccessful"] = []

    # On utilise tqdm pour voir la barre de progression
    # On limite à args.test_num si besoin, sinon on fait tout le dataset
    limit = min(len(data_list), args.test_num) if args.test_num > 0 else len(data_list)
    
    for i in tqdm(range(limit)):
        sentence = data_list[i]['input']
        prompt = prompt_construct(label_list, sentence)
        response = chat(prompt, client)
        
        final_label = "Unsuccessful"
        if response:
            final_label = answer_process(response, label_list)
        
        results[final_label].append(sentence)

    # Sauvegarde
    output_name = f"{args.data}_small_find_labels.json"
    output_file = os.path.join(args.output_path, output_name)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument("--data", type=str, default="arxiv_fine")
    parser.add_argument("--output_path", type=str, default="./generated_labels")
    parser.add_argument("--use_large", action="store_true")
    parser.add_argument("--test_num", type=int, default=100) # Nombre de phrases à classifier
    args = parser.parse_args()
    main(args)