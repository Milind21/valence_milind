import random
import json

with open("conversation_data.json", "r", encoding="utf-8") as file:
        data = json.load(file)

# Function to select a specific percentage of items with each score
def sample_data(data, score_percentage_map):
    result = []
    for score, percentage in score_percentage_map.items():
        # Filter the dictionaries that match the current score
        score_filtered = [item for item in data if int(item['emo']) == score]
        # Determine the number of items to sample
        num_samples = max(1, int(len(score_filtered) * percentage))  # Ensure at least 1 sample if percentage > 0
        # Randomly sample the required number of items
        sampled_items = random.sample(score_filtered, min(num_samples, len(score_filtered)))
        result.extend(sampled_items)
    return result

# Define the percentage for each score
score_percentage_map = {
    1: 0.15,
    2: 0.15,
    3: 0.40,
    4: 0.15,
    5: 0.15
}

# Sample the data
unbiased_data = sample_data(data, score_percentage_map)
file_path = 'unbiased_data.json'
with open(file_path, 'w') as json_file:
    json.dump(unbiased_data, json_file, indent=4)
# Output the sampled data
# print("Sampled Data:")
# for item in unbiased_data:
#     print(item)
