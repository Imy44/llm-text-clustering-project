import random
import os
import json
import argparse
import time

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


# =========================
# CONFIG
# =========================
DEBUG = False


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
def load_dataset(data_path, data, use_large, max_samples):
    file_name = "large.jsonl" if use_large else "small.jsonl"
    data_file = os.path.join(data_path, data, file_name)

    print(f"Use dataset {data_file}")

    data_list = []
    with open(data_file, 'r') as f:
        for line in f:
            data_list.append(json.loads(line))

    if max_samples:
        data_list = data_list[:max_samples]

    print(f"Length of dataset: {len(data_list)}")
    return data_list


def get_label_list(data_list):
    return list(set([d["label"] for d in data_list]))


# =========================
# PROMPTS
# =========================
def prompt_construct_generate_label(sentence_list, current_labels):
    return f"""
You are performing text clustering.

Existing labels:
{current_labels}

For EACH sentence below, assign a label.

RULES:
- If it matches an existing label → use it
- If not → CREATE a new label
- Evaluate EACH sentence independently

Sentences:
{sentence_list}

Output JSON ONLY:
{{
  "results": [
    {{"sentence": "...", "label": "..."}}
  ]
}}
"""


def prompt_construct_merge_label(label_list, target_clusters):
    return f"""
You are merging cluster labels.

OBJECTIVE:
Reduce the number of labels to approximately {target_clusters} clusters.

CRITICAL CONSTRAINT:
- You MUST reduce the total number of labels close to {target_clusters}
- If there are too many labels → aggressively merge similar ones
- If labels are clearly different → keep them separate

STRATEGY:
- Group labels by semantic similarity
- Merge fine-grained labels into broader categories
- Avoid keeping overly specific distinctions

INPUT LABELS:
{label_list}

OUTPUT JSON ONLY:
{{"merged_labels": ["label1", "label2", "..."]}}
"""


def get_sentences(batch):
    return [x["input"] for x in batch]


# =========================
# LABEL GENERATION
# =========================
def label_generation(args, client, data_list):
    all_labels = []

    with open(args.given_label_path, 'r') as f:
        given_labels = json.load(f)[args.data]

    all_labels.extend(given_labels)

    total = len(data_list)

    for i in range(0, total, args.chunk_size):

        progress = round(i / total * 100, 2)
        print(f"Processing {i}/{total} ({progress}%) | labels: {len(all_labels)}")

        batch = data_list[i:i + args.chunk_size]
        sentences = get_sentences(batch)

        prompt = prompt_construct_generate_label(sentences, all_labels)
        response = chat(prompt, client)

        if not response:
            continue

        try:
            response_clean = response.strip()

            if "{" in response_clean:
                response_clean = response_clean[response_clean.find("{"):]
            if "}" in response_clean:
                response_clean = response_clean[:response_clean.rfind("}") + 1]

            parsed = json.loads(response_clean)
            results = parsed.get("results", [])

            for item in results:
                label = item["label"].lower().strip()

                label = label.replace(" ", ".")
                label = label.replace("-", ".")

                if "unknown" in label or "new_label" in label:
                    continue

                if label not in all_labels:
                    all_labels.append(label)

        except Exception:
            continue

    return list(set(all_labels))


# =========================
# MERGE LABELS (CORRIGÉ)
# =========================
def merge_labels(args, all_labels, client, target_clusters):
    print("🔄 Merging labels...")

    prompt = prompt_construct_merge_label(all_labels, target_clusters)
    response = chat(prompt, client)

    if not response:
        print("⚠️ Merge API failed")
        return list(set(all_labels))

    try:
        response_clean = response.strip()

        if "{" in response_clean:
            response_clean = response_clean[response_clean.find("{"):]
        if "}" in response_clean:
            response_clean = response_clean[:response_clean.rfind("}") + 1]

        parsed = json.loads(response_clean)
        merged = parsed.get("merged_labels", [])

        return merged

    except Exception:
        return list(set(all_labels))


# =========================
# SAVE JSON
# =========================
def write_json(args, data, name):
    size = "large" if args.use_large else "small"
    file_name = os.path.join(args.output_path, f"{args.data}_{size}_{name}.json")

    with open(file_name, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved → {file_name}")


# =========================
# MAIN (CORRIGÉ)
# =========================
def main(args):
    start = time.time()

    client = ini_client(args.api_key)

    data_list = load_dataset(
        args.data_path,
        args.data,
        args.use_large,
        args.max_samples
    )

    random.shuffle(data_list)

    true_labels = get_label_list(data_list)
    true_cluster_num = len(true_labels)

    print(f"True cluster num: {true_cluster_num}")

    write_json(args, true_labels, "true_labels")

    # GENERATE
    all_labels = label_generation(args, client, data_list)
    print(f"Generated labels: {len(all_labels)}")

    write_json(args, all_labels, "labels_before_merge")

    # MERGE (CORRIGÉ)
    final_labels = merge_labels(args, all_labels, client, true_cluster_num)
    print(f"Final labels: {len(final_labels)}")

    write_json(args, final_labels, "labels_after_merge")

    print(f"Total time: {time.time() - start:.2f}s")


# =========================
# ARGS
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument("--data", type=str, default="arxiv_fine")
    parser.add_argument("--output_path", type=str, default="./generated_labels")
    parser.add_argument("--given_label_path", type=str, default="./generated_labels/chosen_labels.json")

    parser.add_argument("--use_large", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=15)
    parser.add_argument("--max_samples", type=int, default=1000)

    parser.add_argument("--api_key", type=str, required=True)

    args = parser.parse_args()

    main(args)