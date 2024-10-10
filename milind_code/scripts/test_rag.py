import extract_rag
import json
import random
import sys
import statistics

def main():
    query = input("Enter the converstation:")
    _, score = extract_rag.extract_demo(query,int(5))
    print(f"The query {query} has a satisfaction score of {statistics.mode(score)}")
    
    
if __name__ == "__main__":
    main()