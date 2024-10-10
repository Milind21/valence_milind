import random
import os 
import pandas as pd
import numpy as np
import copy
import sys
import pickle


# HELPER FUNCTIONS 
# (src:https://github.com/sunnweiwei/user-satisfaction-simulation/baselines/svm.py)
def load_pkl(filename):
  with open(filename, 'rb') as filehandle:
    return pickle.load(filehandle)
  
def save_pkl(filename, object):
  with open(filename, 'wb') as filehandle:
    pickle.dump(object, filehandle)
    
# return the mode of annotator scores 
# voting instead of going for mean or some other aggregator    
def get_main_score(scores):
    number = [0, 0, 0, 0, 0]
    for item in scores:
        number[item] += 1
    score = np.argmax(number)
    return score

def load_data(filename):
  raw = [line[:-1] for line in open(filename, encoding='utf-8')]
  data = []
  for line in raw:
      if line == '':
          data.append([]) #add empty to the array if line is blank
      else:
          data[-1].append(line) #add data to the array
  x = []
  emo = []
  act = []
  action_list = {}
  for session in data:  #for each conversation
      his_input_ids = []
      for turn in session: # for each turn split and store
          role, text, action, score = turn.split('\t')
          score = score.split(',')
          action = action.split(',')
          action = action[0]
          if role.upper() == 'USER':
              x.append(' '.join(his_input_ids))              
              emo.append(get_main_score([int(item) - 1 for item in score]))
              action = action.strip()
              if action not in action_list:
                  action_list[action] = len(action_list)
              act.append(action_list[action])
          his_input_ids.append(text.strip())
  action_num = len(action_list)
  # Store the text, score, action idx from dict, and num unique action roles
  data = [x, emo, act, action_num] 
  return data


def main():
  if len(sys.argv) != 3:
        print("Sample CLI execution: preprocess.py inputfile outputfile")
        sys.exit(1)
  ip_path = sys.argv[1]
  op_path = sys.argv[2]
  data = load_data(ip_path)    
  save_pkl(op_path,data)
  
  
if __name__ == "__main__": 
    main()



