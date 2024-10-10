# TODO: zero shot, few shot randomly select demo, vector store and extract relevant
import json
import pickle
import openai
import re
import keys
import weave
import sys
import random
openai.api_key = keys.valence_openai

    
def extract_few_shot_examples(data, scores=[1, 2, 3, 4, 5]):
    few_shot_examples = []
    for score in scores:
        # Filter data by the current score
        filtered_data = [item for item in data if int(item['emo']) == score]
        
        if filtered_data:
            # Randomly select one example from the filtered data
            random_example = random.choice(filtered_data)
            few_shot_examples.append(random_example)
    
    return few_shot_examples

def find_user_satisfaction(text,demo_text,demo_score):
    # System and user prompts as described
    system_prompt = """You are an expert linguistic assistant.
Your task is to label and give a score to conversations for user satisfaction. The score for the user satisfaction are based on a 5-level satisfaction scale.
The scale is as follows: 
(1) Very dissatisfied (the system fails to understand and fulfill users request); 
(2) Dissatisfied (the system understands the request but fails to satisfy it in any way); 
(3) Normal (the system understands users request and either partially satisfies the request or provides information on how the request can be fulfilled); 
(4) Satisfied (the system understands and satisfies the user request, but provides more information than what the user requested or takes extra turns before meeting the request); and 
(5) Very satisfied (the system understands and satisfies the user request completely and efficiently).
You should predict **only the score** and print it as `Score:` followed by the appropriate number.
"""
    demo_prompt = f"""Text 1: {demo_text[0]}
Score 1: {demo_score[0]}
Text 2: {demo_text[1]}
Score 2: {demo_score[1]}
Text 3: {demo_text[2]}
Score 3: {demo_score[2]}
Text 4: {demo_text[3]}
Score 4: {demo_score[3]}
Text 5: {demo_text[4]}
Score 5: {demo_score[4]}
    """
    user_prompt = f"""Text: {text}"""

    # Make the API call
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        # model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": "Use the following examples to score the text. The 5 examples each contain a sample text and score. Every example has a unique score to guide you better."+demo_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # Extract and return the response
    output = response.choices[0].message
    usage = response.usage
    return output, usage


def main():
    with open("../milind_code/ip_data/unbiased_data.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    texts = [item['text'] for item in data]
    gold_labels = [item['emo'] for item in data]
    total_tokens=0
    
    # Store the selected examples in demo_text and demo_score lists
    few_shot_examples = extract_few_shot_examples(data)
    # Extract the text and score for each few-shot example
    demo_text = [item['text'] for item in few_shot_examples]
    demo_score = [item['emo'] for item in few_shot_examples]
    # for i in range(len(data)):
    text = input("Enter the converstation:")
    result_dict = {}
    result_dict["text"] = text
    result,usage = find_user_satisfaction(text,demo_text,demo_score)
    for line in result.content.split('\n'):
      if line.startswith("Score:"):
        result_dict["Score"] = line.split(":")[1].strip()  # Extracting the score

    print(f"You used a total of {usage.total_tokens}")
    print(f"The query {result_dict["text"]} has a satisfaction score of {result_dict["Score"]}")
    
if __name__ == "__main__":
    main()