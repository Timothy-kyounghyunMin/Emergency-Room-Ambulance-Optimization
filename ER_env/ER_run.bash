#!/bin/bash

# Initial seed numbers
seed_numbers=()

for (( i = 21; i <= 30; i++ )); do
  new_number=$((i * 10))
  seed_numbers+=($new_number)
done
# Define an array of model types
model_types=("opt" "naive")

# Create a function to run the Python script with specified arguments
run_script() {
    python ER_env_main.py "$1" "$2"
}

# Loop through seed numbers and model types and run each combination in the background
for seed in "${seed_numbers[@]}"; do
    for model_type in "${model_types[@]}"; do
        run_script "$seed" "$model_type" &
    done
done

# Wait for all background processes to finish
wait