import json
import statistics


with open('conversation_data.json', 'r') as f1:
    data1 = json.load(f1) 
    
def main():
    with open('conversation_data.json', 'r') as f1:
        data = json.load(f1)
    
    score = [int(item["emo"]) for item in data]
    mean_emo = statistics.mean(score)
    median_emo = statistics.median(score)
    mode_emo = statistics.mode(score)
    std_dev_emo = statistics.stdev(score) if len(score) > 1 else 0  # stdev needs more than 1 value
    min_emo = min(score)
    max_emo = max(score)
    print(f"""Mean of the gold labels is {mean_emo}\nMedian of the gold labels is {median_emo}\nMode of the gold labels is {mode_emo}
Std Dev of the gold labels is {std_dev_emo}\nMin of the gold labels is {min_emo}\nMax of the gold labels is {max_emo}\n""")

if __name__=="__main__":
    main()
