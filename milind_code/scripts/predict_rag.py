import extract_rag
import json
import random
import sys
import statistics

def main():
    if len(sys.argv) != 4:
        print("Sample CLI execution: python predict_rag.py gold_path.json pred_path.json num_extract")
        sys.exit(1)
    ip_path = sys.argv[1]
    op_path = sys.argv[2]
    num_extract = sys.argv[3]
    # Read the JSON file
    with open(ip_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    results=[]
    texts = [item['text'] for item in data]
    
    #create short db
    random_indices_500 = random.sample(range(len(texts)), 500)
    for i in range(500):
        result_dict = {}
        text = texts[i]
        # Select 3 examples based on vector search
        _, demo_score = extract_rag.extract_demo(text,int(num_extract))
        result_dict["text"] = text
        result_dict["Score"] = statistics.mode(demo_score)
        results.append(result_dict)
    with open(op_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
if __name__ == "__main__":
  main()