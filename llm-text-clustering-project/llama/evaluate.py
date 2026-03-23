import json
import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import os
import argparse

def cluster_acc(y_true, y_pred):
    if len(y_pred) == 0: return 0
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

def evaluation(y_true, y_pred):
    if len(y_pred) == 0: return 0, 0, 0
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    acc = cluster_acc(y_true, y_pred)
    return nmi, ari, acc

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main(args):
    find_labels_file = os.path.join(args.output_path, f"{args.data}_small_find_labels.json")
    if not os.path.exists(find_labels_file):
        print(f"❌ Fichier introuvable : {find_labels_file}")
        return
        
    find_labels = load_json(find_labels_file)
    dataset_file = os.path.join(args.data_path, args.data, "small.jsonl")
    dataset = load_jsonl(dataset_file)
    
    # Gestion du test_num 0 (tout le dataset)
    limit = len(dataset) if args.test_num <= 0 else args.test_num
    
    tested_sentences = [d['input'] for d in dataset[:limit]]
    true_labels_list = [d['label'] for d in dataset[:limit]]
    
    all_true_unique = sorted(list(set(true_labels_list)))
    true_label_to_id = {name: i for i, name in enumerate(all_true_unique)}
    y_true = np.array([true_label_to_id[l] for l in true_labels_list])
    
    y_pred = []
    missing_count = 0
    
    for sentence in tested_sentences:
        found = False
        for label_name, sentences in find_labels.items():
            if sentence in sentences:
                y_pred.append(label_name)
                found = True
                break
        if not found:
            y_pred.append("Unsuccessful")
            missing_count += 1

    if len(y_pred) == 0:
        print("❌ Erreur : Aucune prédiction n'a pu être associée au dataset.")
        return

    all_pred_unique = sorted(list(set(y_pred)))
    pred_label_to_id = {name: i for i, name in enumerate(all_pred_unique)}
    y_pred_numeric = np.array([pred_label_to_id[l] for l in y_pred])
    
    nmi, ari, acc = evaluation(y_true, y_pred_numeric)
    
    print(f"\n--- Results for {args.data} ({limit} samples) ---")
    print(f"NMI: {nmi:.4f}")
    print(f"ARI: {ari:.4f}")
    print(f"ACC: {acc:.4f}")
    print(f"Clusters prédits: {len(all_pred_unique)} | Clusters réels: {len(all_true_unique)}")
    if missing_count > 0:
        print(f"⚠️ Warning: {missing_count} phrases n'ont pas été retrouvées dans les résultats.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="arxiv_fine")
    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument("--output_path", type=str, default="./generated_labels")
    parser.add_argument("--test_num", type=int, default=0)
    args = parser.parse_args()
    main(args)