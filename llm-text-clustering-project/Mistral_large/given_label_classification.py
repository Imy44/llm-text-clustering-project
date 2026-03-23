import random
import os
import json
import argparse
import time

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


# =========================
# INIT CLIENT
# =========================
def ini_client(api_key):
    return MistralClient(api_key=api_key)


def chat(prompt, client):
    try:
        response = client.chat(
            model="mistral-large-latest",
            messages=[ChatMessage(role="user", content=prompt)]
        )
        return response.choices[0].message.content
    except Exception as e:
        print("API ERROR:", e)
        return None


# =========================
# LOAD DATASET
# =========================
def load_dataset(data_path, data, use_large):
    file_name = "large.jsonl" if use_large else "small.jsonl"
    data_file = os.path.join(data_path, data, file_name)

    print(f"Use dataset {data_file}")

    data_list = []
    with open(data_file, 'r') as f:
        for line in f:
            data_list.append(json.loads(line))

    print(f"Length of dataset: {len(data_list)}")
    return data_list


# =========================
# LOAD GENERATED LABELS
# =========================
def get_predict_labels(output_path, data):
    file_name = f"{data}_small_labels_after_merge.json"
    data_file = os.path.join(output_path, file_name)

    with open(data_file, 'r') as f:
        labels = json.load(f)

    return list(set(labels))


# =========================
# PROMPT
# =========================
def prompt_construct(label_list, sentence):
    return f"""
You are a text classification system.

TASK:
Assign the sentence to ONE label from the list.

IMPORTANT:
- Choose the MOST relevant label
- Do NOT invent new labels
- Do NOT return explanations

LABEL LIST:
{label_list}

SENTENCE:
{sentence}

OUTPUT JSON ONLY:
{{"label_name": "label"}}
"""


# =========================
# PARSE RESPONSE
# =========================
def answer_process(response, label_list):
    try:
        response_clean = response.strip()

        if "{" in response_clean:
            response_clean = response_clean[response_clean.find("{"):]
        if "}" in response_clean:
            response_clean = response_clean[:response_clean.rfind("}") + 1]

        parsed = json.loads(response_clean)

        # 🔥 récupérer toutes les valeurs possibles
        values = [str(v).lower().strip() for v in parsed.values()]

        for val in values:
            if val in label_list:
                return val

    except:
        pass

    return "Unsuccessful"


# =========================
# CLASSIFICATION
# =========================
def known_label_categorize(args, client, data_list, label_list):
    import time

    answer = {label: [] for label in label_list}
    answer["Unsuccessful"] = []

    total = len(data_list)

    for i in range(total):
        sentence = data_list[i]["input"]

        prompt = prompt_construct(label_list, sentence)
        response = chat(prompt, client)

        # 🔥 éviter rate limit
        time.sleep(0.3)

        if response is None:
            final_label = "Unsuccessful"
        else:
            final_label = answer_process(response, label_list)

        answer[final_label].append(sentence)

        # 🔥 affichage progression propre
        progress = round((i + 1) / total * 100, 2)
        print(f"Processing {i+1}/{total} ({progress}%) | label: {final_label}")

        # 🔥 sauvegarde intermédiaire
        if i % 200 == 0 and i != 0:
            print(f"\n💾 Saving checkpoint at {i} samples...")
            write_answer_to_json(args, answer, args.output_path, args.output_file_name)

    print("\n✅ Classification completed.")

    return answer


# =========================
# SAVE
# =========================
def write_answer_to_json(args, answer, output_path, output_name):
    size = "large" if args.use_large else "small"
    file_name = os.path.join(output_path, f"{args.data}_{size}_{output_name}")

    with open(file_name, 'w') as f:
        json.dump(answer, f, indent=2)

    print(f"Saved → {file_name}")


# =========================
# SUMMARY
# =========================
def describe_final_output(answer):
    for key in answer:
        print(f"{key}: {len(answer[key])}")


# =========================
# MAIN
# =========================
# =========================
# MAIN
# =========================
def main(args):
    start = time.time()

    client = ini_client(args.api_key)

    data_list = load_dataset(
        args.data_path,
        args.data,
        args.use_large
    )

    # 🔥 LIMITATION À 1000 PHRASES
    data_list = data_list[:1000]
    print(f"Dataset truncated to: {len(data_list)} samples")

    label_list = get_predict_labels(args.output_path, args.data)
    print(f"Number of labels: {len(label_list)}")

    answer = known_label_categorize(
        args,
        client,
        data_list,
        label_list
    )

    # supprimer labels vides
    answer = {k: v for k, v in answer.items() if len(v) > 0}

    write_answer_to_json(args, answer, args.output_path, args.output_file_name)

    print("\nFinal distribution:")
    describe_final_output(answer)

    print(f"\nTotal time: {time.time() - start:.2f}s")


# =========================
# ARGS
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument("--data", type=str, default="arxiv_fine")
    parser.add_argument("--output_path", type=str, default="./generated_labels")
    parser.add_argument("--output_file_name", type=str, default="classification.json")

    parser.add_argument("--use_large", action="store_true")
    parser.add_argument("--api_key", type=str, required=True)

    args = parser.parse_args()

    main(args)