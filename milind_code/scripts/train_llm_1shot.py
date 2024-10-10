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

def load_pkl(filename):
  with open(filename, 'rb') as filehandle:
    return pickle.load(filehandle)
  
def save_pkl(filename, object):
  with open(filename, 'wb') as filehandle:
    pickle.dump(object, filehandle)
    


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
You should predict only the score which is a number and print it as Score.
"""
    demo_prompt = f"""Text: {demo_text}
Score: {demo_score}
    """
    user_prompt = f"""Text: {text}"""

    # Make the API call
    response = openai.chat.completions.create(
        # model="gpt-3.5-turbo",
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": "Use the following example to score the text"+demo_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # Extract and return the response
    output = response.choices[0].message
    return output


def main():
    if len(sys.argv) != 3:
        print("Sample CLI execution: python train_llm_1shot.py gold_path.json pred_path.json")
        sys.exit(1)
    ip_path = sys.argv[1]
    op_path = sys.argv[2]
    # Read the JSON file
    with open(ip_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    demo = random.choice(data)
    results = []
    # for i in range(len(data)):
    for i in range(100):
        result_dict = {}
        text=data[i]["text"]
        result_dict["text"] = text
        result = find_user_satisfaction(text,demo["text"],demo["emo"])
        # Extract relevant information
        for line in result.content.split('\n'):
            if line.startswith("Score:"):
                result_dict["Score"] = line.split(":")[1].strip()  # Extracting the score
            # elif line.startswith("Label:"):
                # result_dict["Label"] = line.split(":")[1].strip()  # Extracting the label
            # elif line.startswith("Explanation:"):
                # Extract the entire explanation (it might span multiple lines)
                # explanation = line.split(":")[1].strip()
                # explanation_continued = " ".join(result.content.split("\n")[result.content.split("\n").index(line) + 1:]).strip()
                # result_dict["Explanation"] = explanation + " " + explanation_continued
        results.append(result_dict)
    # Print the resulting dictionary
    with open(op_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()