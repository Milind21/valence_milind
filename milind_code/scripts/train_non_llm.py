# TODO: Come up with some heuristic to find overall satsifaction


import random
import os 
import pandas as pd
import numpy as np
import copy
import sys
import pickle
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from spearman import spearman
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
import wandb


def log_wandb_search(cv_results_, n_iter):
  for i in range(n_iter):
  # Log each combination of hyperparameters tried
    wandb.log({
    'mean_test_custom_score': cv_results_['mean_test_score'][i],
    'std_test_custom_score': cv_results_['std_test_score'][i],
    'rank_test_custom_score': cv_results_['rank_test_score'][i]
    })

def load_pkl(filename):
  with open(filename, 'rb') as filehandle:
    return pickle.load(filehandle)
  
def save_pkl(filename, object):
  with open(filename, 'wb') as filehandle:
    pickle.dump(object, filehandle)

def data_vectorizer(data):
  x, emo, act, action_num = data
  train_x, test_x, train_act, test_act = train_test_split(x, emo, test_size=0.2, random_state=2)
  # Builiding TF-IDF 
  vectorizer = TfidfVectorizer() 
  train_feature = vectorizer.fit_transform(train_x)
  test_feature = vectorizer.transform(test_x)
  return train_feature, test_feature, train_act, test_act
  
def train_lr(data):
  train_feature, test_feature, train_act, test_act = data_vectorizer(data)
  custom_metric = make_scorer(spearman, greater_is_better=True)
  best_model = LogisticRegression()
  # setup session to track using weights and biases
  wandb.init(project="valence", name = "LR")
  best_model.fit(train_feature,train_act)
  pred = best_model.predict(test_feature)
  gold_label = test_act
  # Calculating the unweighted average recall since dataset is has multi-class prediction
  recall = [[0,0] for _ in range(5)]
  for p, l in zip(pred, gold_label):
    recall[l][1] += 1
    recall[l][0] += int(p==l)
  recall_val = [item[0]/max(item[1],1) for item in recall]
  UAR = sum(recall_val)/len(recall_val)
  kappa = cohen_kappa_score(pred,gold_label)
  rho = np.absolute(spearman(gold_label,pred))
  print(f"UAR:{UAR}\nkappa:{kappa}\nrho:{rho}\n")
  wandb.log({
    "Unweighted Average Recall" : UAR,
    "Cohen's Kappa" : kappa,
    "Spearman's rho" : rho
  })
  save_pkl("./models/lr.pkl", best_model)

def train_rf(data):
  train_feature, test_feature, train_act, test_act = data_vectorizer(data)
  custom_metric = make_scorer(spearman, greater_is_better=True)
  model = RandomForestClassifier(random_state=2)
  param_grid = {
        'n_estimators': [5,  10, 20, 50, 100, 200, 500],
        'max_depth': [1, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [2, 4, 8, 16 ],
        'max_features': ['sqrt', 'log2']
    }
  # setup session to track using weights and biases
  wandb.init(project="valence", name = "RF Finetuning 2", config = param_grid)
  # wandb.init(project="valence", name = "RF Baseline")
  cv = KFold(n_splits=5)
  random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=cv, n_jobs=-1, scoring=custom_metric, verbose=3)
  random_search.fit(train_feature, train_act)
  # # adding log to wandb
  log_wandb_search(random_search.cv_results_, random_search.n_iter)
  print("Best Parameters:", random_search.best_params_)
  wandb.log({"Best n_estimators": random_search.best_params_["n_estimators"],
             "Best max_depth": random_search.best_params_["max_depth"],
             "Best min_samples_split": random_search.best_params_["min_samples_split"],
             "Best min_samples_leaf": random_search.best_params_["min_samples_leaf"]
             }) #log best parameters
  best_model = random_search.best_estimator_
  best_model.fit(train_feature,train_act)
  pred = best_model.predict(test_feature)
  gold_label = test_act
  
  # Calculating the unweighted average recall since dataset is has multi-class prediction
  recall = [[0,0] for _ in range(5)]
  for p, l in zip(pred, gold_label):
    recall[l][1] += 1
    recall[l][0] += int(p==l)
  recall_val = [item[0]/max(item[1],1) for item in recall]
  UAR = sum(recall_val)/len(recall_val)
  kappa = cohen_kappa_score(pred,gold_label)
  rho = spearman(pred, gold_label)
  print(f"UAR:{UAR}\nkappa:{kappa}\nrho:{rho}\n")
  wandb.log({
    "Unweighted Average Recall" : UAR,
    "Cohen's Kappa" : kappa,
    "Spearman's rho" : rho
  })
  save_pkl("./models/rf.pkl", best_model)


def train_svc(data):
  train_feature, test_feature, train_act, test_act = data_vectorizer(data)
  custom_metric = make_scorer(spearman, greater_is_better=True)
  model = SVC()
  param_grid = {
        'C': np.logspace(-3, 2, 6),
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': np.logspace(-3, 2, 6)
    }
  # setup session to track using weights and biases
  wandb.init(project="valence", name = "SVC Finetuning ", config = param_grid)
  # wandb.init(project="valence", name = "SVC Baseline")
  cv = KFold(n_splits=5)
  random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=cv, n_jobs=-1, scoring=custom_metric, verbose=3)
  random_search.fit(train_feature, train_act)
  # adding log to wandb
  log_wandb_search(random_search.cv_results_, random_search.n_iter)
  print("Best Parameters:", random_search.best_params_)
  wandb.log({"Best C": random_search.best_params_["C"],
             "Best gamma": random_search.best_params_["gamma"]
             }) #log best parameters
  best_model = random_search.best_estimator_
  best_model.fit(train_feature,train_act)
  pred = best_model.predict(test_feature)
  gold_label = test_act
  
  # Calculating the unweighted average recall since dataset is has multi-class prediction
  recall = [[0,0] for _ in range(5)]
  for p, l in zip(pred, gold_label):
    recall[l][1] += 1
    recall[l][0] += int(p==l)
  recall_val = [item[0]/max(item[1],1) for item in recall]
  UAR = sum(recall_val)/len(recall_val)
  kappa = cohen_kappa_score(pred,gold_label)
  rho = spearman(pred, gold_label)
  print(f"UAR:{UAR}\nkappa:{kappa}\nrho:{rho}\n")
  wandb.log({
    "Unweighted Average Recall" : UAR,
    "Cohen's Kappa" : kappa,
    "Spearman's rho" : rho
  })
  save_pkl("./models/svc.pkl", best_model)


def train_xgb(data):
  train_feature, test_feature, train_act, test_act = data_vectorizer(data)
  custom_metric = make_scorer(spearman, greater_is_better=True)
  model = XGBClassifier()
  param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': np.logspace(-3, 0, 5),
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.3, 0.5]
    }
  # setup session to track using weights and biases
  wandb.init(project="valence", name = "XGB Finetuning ", config = param_grid)
  # wandb.init(project="valence", name = "XGB Baseline")
  cv = KFold(n_splits=5)
  random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=cv, n_jobs=-1, scoring=custom_metric, verbose=3)
  random_search.fit(train_feature, train_act)
  # adding log to wandb
  log_wandb_search(random_search.cv_results_, random_search.n_iter)
  print("Best Parameters:", random_search.best_params_)
  wandb.log({"Best n_estimators": random_search.best_params_["n_estimators"],
             "Best max_depth": random_search.best_params_["max_depth"],
             "Best learning_rate": random_search.best_params_["learning_rate"],
             "Best subsample": random_search.best_params_["subsample"],
             "Best colsample_bytree": random_search.best_params_["colsample_bytree"],
             "Best gamma": random_search.best_params_["gamma"]
             }) #log best parameters
  best_model = random_search.best_estimator_
  best_model.fit(train_feature,train_act)
  pred = best_model.predict(test_feature)
  gold_label = test_act
  
  # Calculating the unweighted average recall since dataset is has multi-class prediction
  recall = [[0,0] for _ in range(5)]
  for p, l in zip(pred, gold_label):
    recall[l][1] += 1
    recall[l][0] += int(p==l)
  recall_val = [item[0]/max(item[1],1) for item in recall]
  UAR = sum(recall_val)/len(recall_val)
  kappa = cohen_kappa_score(pred,gold_label)
  rho = spearman(pred, gold_label)
  print(f"UAR:{UAR}\nkappa:{kappa}\nrho:{rho}\n")
  wandb.log({
    "Unweighted Average Recall" : UAR,
    "Cohen's Kappa" : kappa,
    "Spearman's rho" : rho
  })
  save_pkl("./models/xgb.pkl", best_model)


def main():
  if len(sys.argv) != 3:
    print("Sample CLI execution: non_llm.py data_pickle model_name\nNonLLM Based Models: LR, RF, SVC, XGB")
    sys.exit(1)
  data_pickle_path = sys.argv[1]
  model_name = sys.argv[2]
  data = load_pkl(data_pickle_path)
  if model_name == "RF":
    train_rf(data=data)
  elif model_name == "LR":
    train_lr(data=data)
  elif model_name == "SVC":
    train_svc(data=data)
  elif model_name == "XGB":
    train_xgb(data=data)  
  else:
    print("Invalid model name. Please choose from: LR, RF, SVC, XGB")
    sys.exit(1)
    
  wandb.finish()

if __name__ == "__main__": 
    main()