import json
import statistics

def parse_conversation_file(filename):
    # Step 1: Parse the input file and structure the conversation data
    raw = [line[:-1] for line in open(filename, encoding='utf-8')]
    data = []
    for line in raw:
        if line == '':
            data.append([]) #add empty to the array if line is blank
        else:
            data[-1].append(line)  # Append the line to the current session
    
    x = []
    emo = []

    # Step 2: Extract relevant parts of each conversation turn (role, text, action, score)
    for session in data:
        his_input_ids = []
        for turn in session:
            role, text, _ , score = turn.split('\t')
            score = score.split(',')
            # Store user and system conversation data
            his_input_ids.append((role, text, score))  # Keep track of role, text, and scores
        x.append(his_input_ids)
    return x

def generate_json_from_conversations(conversations):
    json_data = []

    # Step 3: Pair user-system conversations and calculate mode of the scores
    for session in conversations:
        for i in range(0, len(session) - 1, 2):
            user_turn = session[i]
            system_turn = session[i + 1]

            if user_turn[0] == 'USER' and system_turn[0] == 'SYSTEM':
                # Combine user and system conversation text
                text_pair = f"USER: {user_turn[1]} SYSTEM: {system_turn[1]}"
                
                # Combine the scores and calculate the mode
                combined_scores = user_turn[2] + system_turn[2]
                mode_score = statistics.mode(combined_scores)
                
                # Add to JSON data
                json_data.append({"text": text_pair, "emo": mode_score})
    
    return json_data

def save_json(json_data, output_file):
    # Step 4: Save the conversation data to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

if __name__ == "__main__":
    # Input conversation file
    input_file = 'SGD.txt'
    output_file = 'conversation_data.json'
    
    # Step 1: Parse the conversation file
    conversations = parse_conversation_file(input_file)
    
    # Step 2: Generate the required JSON structure
    json_data = generate_json_from_conversations(conversations)
    
    # Step 3: Save the JSON output
    save_json(json_data, output_file)
    
    print(f"JSON data saved to {output_file}")
