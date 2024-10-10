#!/bin/bash
models=("RF" "LR" "SVC" "XGB")

for model in "${models[@]}"
do
  echo "Running model: $model"
  python ./scripts/train_non_llm.py ./ip_data/data.pkl "$model"
  # Check if the command ran successfully
  if [ $? -eq 0 ]; then
    echo "$model ran successfully"
  else
    echo "$model failed to run"
    exit 1
  fi
done
echo "All models have been processed."
