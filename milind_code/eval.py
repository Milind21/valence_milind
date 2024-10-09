from spearman import spearman
from sklearn.metrics import cohen_kappa_score
import json
import numpy as np
import sys

def calculate_llm_metrics(gold_path, pred_path):
    with open(gold_path, 'r') as f1, open(pred_path, 'r') as f2:
        data1 = json.load(f1)  # List of dictionaries from file1
        data2 = json.load(f2)  # List of dictionaries from file2
    # print(data1)
    # print(data2)
    # Initialize empty lists to store gold and pred labels
    gold_label = []
    pred = []

    # Extract 'emo' (gold labels) from file1
    for entry in data1:
        if 'emo' in entry:
            gold_label.append(entry['emo'])  # Add 'emo' to the gold_labels list

    # Extract 'emo' (pred labels) from file2
    for entry in data2:
        if 'Score' in entry:
            pred.append(entry['Score']) 
    gold_label=gold_label[:100]
    gold_label = [int(item) for item in gold_label]
    pred = [int(item) for item in pred]
    recall = [[0,0] for _ in range(5)]
    for p, l in zip(pred, gold_label):
        recall[l][1] += 1
        recall[l][0] += int(p==l)
    recall_val = [item[0]/max(item[1],1) for item in recall]
    UAR = sum(recall_val)/len(recall_val)
    kappa = np.absolute(cohen_kappa_score(pred,gold_label))
    rho = np.absolute(spearman(gold_label,pred))
    print(f"UAR:{UAR}\nkappa:{kappa}\nrho:{rho}\n")
    
def main():
    if len(sys.argv) != 3:
        print("Sample CLI execution: python eval.py gold_path.json pred_path.json")
        sys.exit(1)
    gold_path = sys.argv[1]
    pred_path = sys.argv[2]
    calculate_llm_metrics(gold_path,pred_path)

if __name__ == "__main__":
    main()
