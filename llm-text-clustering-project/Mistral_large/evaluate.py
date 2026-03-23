import argparse
import os
import json
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment


# =========================
# LOAD DATA
# =========================
def load_data(data_path, data, use_large):
    data_file = os.path.join(data_path, data, "large.jsonl") if use_large else os.path.join(data_path, data, "small.jsonl")
    
    data_list = []
    with open(data_file, 'r') as f:
        for line in f:
            data_list.append(json.loads(line))
    
    return data_list


def load_predict_data(data_path, file_name):
    data_file = os.path.join(data_path, file_name)
    
    with open(data_file, 'r') as f:
        data_dict = json.load(f)
    
    return data_dict


# =========================
# LABEL EXTRACTION
# =========================
def get_labels(data_list):
    return [data["label"] for data in data_list]


# 🔥 VERSION OPTIMISÉE (O(n))
def get_predict_labels(label_data_list, predict_data_dict):
    # construire mapping sentence → label
    sentence_to_label = {}

    for label, sentences in predict_data_dict.items():
        for s in sentences:
            sentence_to_label[s] = label

    predict_labels = []

    for data in label_data_list:
        sentence = data["input"]
        label = sentence_to_label.get(sentence, "Unsuccessful")
        predict_labels.append(label)

    return predict_labels


# =========================
# CONVERSION LABELS → IDS
# =========================
def convert_label_to_ids(labels):
    unique_labels = list(set(labels))
    n_clusters = len(unique_labels)

    label_map = {l: i for i, l in enumerate(unique_labels)}
    label_ids = [label_map[l] for l in labels]

    print(f"Length of labels: {len(labels)}")
    print(f"Number of Clusters: {n_clusters}")

    return np.asarray(label_ids), n_clusters


# =========================
# METRICS
# =========================
def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w


def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc


def clustering_score(y_true, y_pred):
    return {
        'ACC': clustering_accuracy_score(y_true, y_pred),
        'ARI': adjusted_rand_score(y_true, y_pred),
        'NMI': normalized_mutual_info_score(y_true, y_pred)
    }


# =========================
# MAIN
# =========================
def main(args):
    # 🔥 charger dataset
    label_data_list = load_data(args.data_path, args.data, args.use_large)

    # 🔥 IMPORTANT : aligner avec classification (1000 samples)
    label_data_list = label_data_list[:1000]

    labels = get_labels(label_data_list)
    print(f"Total ground truth labels: {len(labels)}")

    # 🔥 charger prédictions
    predict_data_dict = load_predict_data(args.predict_file_path, args.predict_file)

    predict_labels = get_predict_labels(label_data_list, predict_data_dict)

    print(f"Total predicted labels: {len(predict_labels)}")

    # =========================
    # 🔥 FILTRER "Unsuccessful"
    # =========================
    filtered = [(t, p) for t, p in zip(labels, predict_labels) if p != "Unsuccessful"]

    labels = [t for t, p in filtered]
    predict_labels = [p for t, p in filtered]

    print(f"After filtering Unsuccessful: {len(labels)} samples")

    # =========================
    # CONVERT TO IDS
    # =========================
    print("\nGround truth labels:")
    y_true, cluster_true = convert_label_to_ids(labels)

    print("\nPredicted labels:")
    y_pred, cluster_pred = convert_label_to_ids(predict_labels)

    # =========================
    # METRICS
    # =========================
    score = clustering_score(y_true=y_true, y_pred=y_pred)

    print("\nFinal Scores:")
    print(score)


# =========================
# ARGS
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument("--data", type=str, default="arxiv_fine")
    parser.add_argument("--use_large", action="store_true")

    parser.add_argument("--predict_file_path", type=str, default="./generated_labels/")
    parser.add_argument("--predict_file", type=str, required=True)

    args = parser.parse_args()

    main(args)